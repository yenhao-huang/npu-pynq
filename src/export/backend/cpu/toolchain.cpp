//===- toolchain.cpp - CPU code for one target ----------------------------===//
//
// Upstream MLIR passes, in process, from linalg on tensors to the LLVM
// dialect:
//
//   loop interchange for unit-stride innermost loops (interchangeForUnitStride)
//   -> bufferize (function boundaries included) -> results become out-params
//   -> deallocation -> loops -> LLVM dialect, index width of the target
//
// then LLVM IR, opt (kCanonicalPasses, then kVectorizePasses) and llc for the
// target CPU, and a shared library linked with the MLIR runtime support the
// generated code may call (memrefCopy). Between the two opt runs, every
// innermost loop that carries no floating-point value asks to be vectorized
// (vectorizeHints), so loops become NEON code on the Cortex-A9. Public
// functions keep MLIR's C interface (_mlir_ciface_<name>, memref
// descriptors); __delegate__ stays undefined for the runtime to provide.
//
//===----------------------------------------------------------------------===//

#include "toolchain.h"

#include "kernel_probes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace backend {
namespace {

// memref.copy between non-contiguous memrefs lowers to a call of this
// function (normally MLIR's c_runner_utils). Freestanding, so the same source
// compiles for every target without a sysroot.
constexpr const char *kRuntimeSupport = R"c(
typedef __INTPTR_TYPE__ intptr_t;
typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
typedef struct { intptr_t rank; void *descriptor; } unranked_memref;

/* Descriptor layout: { allocated, aligned, offset, sizes[rank], strides[rank] }.
   Ranks above 8 are not supported: such a copy is silently skipped. */
void memrefCopy(intptr_t elem_size, unranked_memref *src, unranked_memref *dst) {
  intptr_t rank = src->rank;
  char *s_base = ((char **)src->descriptor)[1];
  char *d_base = ((char **)dst->descriptor)[1];
  intptr_t *s_meta = (intptr_t *)((char **)src->descriptor + 2);
  intptr_t *d_meta = (intptr_t *)((char **)dst->descriptor + 2);
  intptr_t s_off = s_meta[0], d_off = d_meta[0];
  intptr_t *sizes = s_meta + 1, *s_strides = s_meta + 1 + rank, *d_strides = d_meta + 1 + rank;
  if (rank == 0) {
    memcpy(d_base + d_off * elem_size, s_base + s_off * elem_size, (size_t)elem_size);
    return;
  }
  for (intptr_t i = 0; i < rank; ++i)
    if (sizes[i] == 0) return;
  intptr_t index[8] = {0};
  if (rank > 8) return;
  int contiguous = s_strides[rank - 1] == 1 && d_strides[rank - 1] == 1;
  intptr_t run = contiguous ? sizes[rank - 1] : 1;
  for (;;) {
    intptr_t so = s_off, d = d_off;
    for (intptr_t i = 0; i < rank; ++i) {
      so += index[i] * s_strides[i];
      d += index[i] * d_strides[i];
    }
    memcpy(d_base + d * elem_size, s_base + so * elem_size, (size_t)(run * elem_size));
    intptr_t axis = contiguous ? rank - 2 : rank - 1;
    for (; axis >= 0; --axis) {
      if (++index[axis] < sizes[axis]) break;
      index[axis] = 0;
    }
    if (axis < 0) return;
  }
}
)c";

// The IR passes before llc, in two opt runs with vectorizeHints between them.
// The first makes the loops what the loop vectorizer needs: canonical and
// rotated, invariants hoisted (accumulators promoted to registers), and
// invariant branches such as the cf.assert of a fused quantization unswitched
// out of them. The second vectorizes and cleans up. Not -O3: prefill and
// decode are single functions of about 100k instructions and 10k loops, and
// -O3 spends 37 minutes on them (-O1 9 minutes) in loop passes that run once
// per loop over the whole function. These take under a minute together.
constexpr const char *kCanonicalPasses =
    "function(sroa,early-cse,instcombine,simplifycfg,loop-simplify,lcssa,"
    "loop(loop-rotate),loop-mssa(licm,simple-loop-unswitch,licm),loop(indvars))";
constexpr const char *kVectorizePasses = "function(loop-vectorize,early-cse,instcombine,simplifycfg)";

// Bufferization, and what follows it to the LLVM dialect. --profile-kernels
// runs the kernel probes before the first part and between the two.
std::string bufferization() {
  return "one-shot-bufferize{bufferize-function-boundaries "
         "function-boundary-type-conversion=identity-layout-map},"
         "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true},"
         "canonicalize,buffer-deallocation-pipeline,convert-bufferization-to-memref";
}

std::string toLlvm(unsigned indexBits) {
  std::string ib = "index-bitwidth=" + std::to_string(indexBits);
  return "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
         "convert-scf-to-cf,finalize-memref-to-llvm{" + ib + "},"
         "convert-math-to-llvm,convert-arith-to-llvm{" + ib + "},"
         "convert-cf-to-llvm{" + ib + "},convert-func-to-llvm{" + ib + "},"
         "convert-index-to-llvm{" + ib + "},reconcile-unrealized-casts";
}

std::string pipeline(unsigned indexBits) {
  // The pass manager is already anchored on builtin.module.
  //
  // These options fix the ABI that the runtime calls:
  //   identity-layout-map           public memrefs stay plain, as the runtime's descriptors assume
  //   buffer-results-to-out-params  results become caller-owned arguments, written directly
  //   index-bitwidth                matches the target's pointer size, used by descriptor fields
  //
  // Graph-level cleanup and elementwise fusion already ran in pynqo-middle-end.
  // The canonicalize below cleans up bufferization, which is per target.
  return bufferization() + "," + toLlvm(indexBits);
}

// Run `passes` (a pipeline string) on `module`.
bool runPipeline(ModuleOp module, const std::string &passes, std::string &error) {
  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(passes, pm))) {
    error = "bad CPU pass pipeline";
    return false;
  }
  if (failed(pm.run(module))) {
    error = "lowering failed";
    return false;
  }
  return true;
}

bool run(const std::string &program, std::vector<std::string> args, std::string &error) {
  std::vector<llvm::StringRef> argv;
  argv.push_back(program);
  for (const std::string &a : args)
    argv.push_back(a);
  std::string message;
  int rc = llvm::sys::ExecuteAndWait(program, argv, std::nullopt, {}, 0, 0, &message);
  if (rc != 0) {
    error = llvm::sys::path::filename(program).str() + " failed (" + std::to_string(rc) +
            ") " + message;
    return false;
  }
  return true;
}

bool writeText(const std::string &path, llvm::StringRef text, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  os << text;
  return true;
}

// How many operands of `op` a loop over `dim` reads or writes with a stride
// other than one: those whose indexing map uses `dim` anywhere but as its
// whole last result. An operand that does not use `dim` is invariant in it.
unsigned stridedOperands(linalg::LinalgOp op, unsigned dim) {
  unsigned strided = 0;
  for (AffineMap map : op.getIndexingMapsArray()) {
    unsigned n = map.getNumResults();
    bool used = false, unitStride = false;
    for (unsigned r = 0; r < n; ++r) {
      AffineExpr e = map.getResult(r);
      if (!e.isFunctionOfDim(dim))
        continue;
      used = true;
      auto d = dyn_cast<AffineDimExpr>(e);
      unitStride = r == n - 1 && d && d.getPosition() == dim;
      if (!unitStride)
        break;
    }
    if (used && !unitStride)
      ++strided;
  }
  return strided;
}

// The dimension to make innermost: the one with the fewest strided operands,
// the current innermost (the last) on a tie. Too short a loop to fill a
// vector is not worth moving inward.
unsigned bestInnermost(linalg::LinalgOp op) {
  unsigned loops = op.getNumLoops();
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  unsigned best = loops - 1, fewest = stridedOperands(op, best);
  for (unsigned d = 0; d + 1 < loops; ++d) {
    if (!ShapedType::isDynamic(ranges[d]) && ranges[d] < 4)
      continue;
    unsigned strided = stridedOperands(op, d);
    if (strided < fewest) {
      best = d;
      fewest = strided;
    }
  }
  return best;
}

// convert-linalg-to-loops nests the loops in iterator order, so a linalg
// op's innermost loop is its last dimension. Make it the dimension with the
// fewest strided operands (bestInnermost), so that LLVM's loop vectorizer sees
// unit-stride or invariant accesses, and work that does not depend on it
// leaves the loop. Named ops (linalg.matmul) are generalized first.
//
//   attention     Q·Kᵀ reduces over the head dimension with K stored
//                 [head_dim, seq], P·V over the sequence with V stored
//                 [seq, head_dim]: both read with a stride; with the output's
//                 last dimension innermost every access is unit-stride or
//                 invariant, and the RoPE fused into Q·Kᵀ is computed once
//                 per element of Q instead of once per key.
//   INT8 linear   matmuls reduce over K with B stored [K, N]; the activation
//                 quantization fused into them (and the SiLU before it)
//                 becomes invariant in N.
//
// The other loops keep their order, so every output element still sums its
// terms in the same order: the result is bit-identical.
void interchangeForUnitStride(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  SmallVector<linalg::LinalgOp> ops;
  module.walk([&](linalg::LinalgOp op) {
    if (op.getNumLoops() >= 2 && op.hasPureTensorSemantics() &&
        bestInnermost(op) != op.getNumLoops() - 1)
      ops.push_back(op);
  });
  for (linalg::LinalgOp op : ops) {
    unsigned loops = op.getNumLoops(), best = bestInnermost(op);
    rewriter.setInsertionPoint(op);
    auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
    if (!generic) {
      DictionaryAttr attrs = op->getDiscardableAttrDictionary();
      FailureOr<linalg::GenericOp> g = linalg::generalizeNamedOp(rewriter, op);
      if (failed(g))
        continue;
      generic = *g;
      generic->setDiscardableAttrs(attrs); // pynqo.kernel_id
    }
    SmallVector<unsigned> order;
    for (unsigned d = 0; d < loops; ++d)
      if (d != best)
        order.push_back(d);
    order.push_back(best);
    (void)linalg::interchangeGenericOp(rewriter, generic, order);
  }
}

// Let LLVM vectorize the CPU code for targets whose SIMD unit is not fully
// IEEE 754, such as ARMv7 NEON (denormals flush to zero): there LLVM's loop
// vectorizer refuses any loop with a floating-point operation not marked
// `fast`. llvm.loop.vectorize.enable on a loop lifts that veto; legality
// checks and the cost model still decide whether and how wide.
//
// The hint also lets the vectorizer reorder a floating-point reduction (as
// `#pragma clang loop vectorize(enable)` does), and W8A8 static quantization
// amplifies that reordering into visibly different logits. So a loop that
// carries a floating-point value from one iteration to the next (a phi of a
// floating-point type in its header: a float reduction, once the first opt run
// has promoted the accumulator) gets no hint and stays scalar and in order.
// interchangeForUnitStride has already turned the hot reductions into loops
// over an output dimension, which carry nothing.
//
// No fast-math flag is set either (`fast` would also be wrong here: its `ninf`
// makes the -inf of attention masks and max reductions poison). The results
// differ from scalar code only where NEON flushes a denormal to zero.
void vectorizeHints(llvm::Module &m) {
  llvm::LLVMContext &ctx = m.getContext();
  for (llvm::Function &f : m) {
    if (f.isDeclaration())
      continue;
    llvm::DominatorTree dt(f);
    llvm::LoopInfo li(dt);
    for (llvm::Loop *loop : li.getLoopsInPreorder()) {
      if (!loop->isInnermost())
        continue;
      bool floatRecurrence = false;
      for (llvm::PHINode &phi : loop->getHeader()->phis())
        floatRecurrence |= phi.getType()->isFPOrFPVectorTy();
      if (floatRecurrence)
        continue;
      llvm::SmallVector<llvm::BasicBlock *, 2> latches;
      loop->getLoopLatches(latches);
      llvm::Metadata *enable[] = {
          llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::getTrue(ctx))};
      llvm::Metadata *ops[] = {nullptr, llvm::MDNode::get(ctx, enable)};
      llvm::MDNode *id = llvm::MDNode::getDistinct(ctx, ops);
      id->replaceOperandWith(0, id);
      for (llvm::BasicBlock *latch : latches)
        latch->getTerminator()->setMetadata(llvm::LLVMContext::MD_loop, id);
    }
  }
}

// Read `in` (bitcode), add vectorizeHints, write `out`.
bool hintBitcode(const std::string &in, const std::string &out, std::string &error) {
  llvm::LLVMContext ctx;
  llvm::SMDiagnostic diag;
  std::unique_ptr<llvm::Module> m = llvm::parseIRFile(in, diag, ctx);
  if (!m) {
    error = "cannot read " + in + ": " + diag.getMessage().str();
    return false;
  }
  vectorizeHints(*m);
  std::error_code ec;
  llvm::raw_fd_ostream os(out, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error = "cannot write " + out + ": " + ec.message();
    return false;
  }
  llvm::WriteBitcodeToFile(*m, os);
  return true;
}

} // namespace

LogicalResult codegen(ModuleOp module, const CpuTarget &target, const std::string &dir,
                      const std::string &llvmBin, bool keepIr, bool profileKernels,
                      std::string &error) {
  llvm::sys::fs::create_directories(dir);
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isPrivate())
      fn->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
  interchangeForUnitStride(module);

  if (!profileKernels) {
    if (!runPipeline(module, pipeline(target.indexBits), error)) {
      error = "lowering to the LLVM dialect failed for " + target.model;
      return failure();
    }
  } else {
    // kernel_probes.cpp: kernels on tensors, then the copies bufferization
    // left outside them.
    std::vector<KernelEntry> table;
    if (failed(probeKernels(module, table))) {
      error = "probing the kernels failed";
      return failure();
    }
    if (!runPipeline(module, bufferization(), error)) {
      error = "bufferization failed for " + target.model;
      return failure();
    }
    probeCopies(module, table);
    if (!runPipeline(module, toLlvm(target.indexBits), error)) {
      error = "lowering to the LLVM dialect failed for " + target.model;
      return failure();
    }
    if (!writeKernelTable(dir + "/kernels.json", table, error))
      return failure();
  }
  if (keepIr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(dir + "/model.llvm.mlir", ec, llvm::sys::fs::OF_Text);
    module->print(os, OpPrintingFlags().elideLargeElementsAttrs(16));
  }
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    error = "translation to LLVM IR failed";
    return failure();
  }
  std::string ll = dir + "/model.ll", canonical = dir + "/model.canonical.bc",
              hinted = dir + "/model.hinted.bc", bc = dir + "/model.opt.bc", obj = dir + "/model.o";
  std::string rtC = dir + "/runtime_support.c", rtObj = dir + "/runtime_support.o";
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(ll, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      error = "cannot write " + ll;
      return failure();
    }
    llvmModule->print(os, nullptr);
  }
  if (!writeText(rtC, kRuntimeSupport, error))
    return failure();

  // opt and llc see the same target, so the vectorizer's cost model is the
  // target CPU's.
  std::vector<std::string> targetFlags = {"-mtriple=" + target.triple};
  if (!target.cpu.empty())
    targetFlags.push_back("-mcpu=" + target.cpu);
  if (!target.features.empty())
    targetFlags.push_back("-mattr=" + target.features);
  std::vector<std::string> first = targetFlags, second = targetFlags;
  first.insert(first.end(), {"-passes=" + std::string(kCanonicalPasses), ll, "-o", canonical});
  second.insert(second.end(), {"-passes=" + std::string(kVectorizePasses), hinted, "-o", bc});
  if (!run(llvmBin + "/opt", first, error) || !hintBitcode(canonical, hinted, error) ||
      !run(llvmBin + "/opt", second, error))
    return failure();
  std::vector<std::string> llc = targetFlags;
  llc.insert(llc.end(), {"-relocation-model=pic", "-O3", "-filetype=obj", bc, "-o", obj});
  if (!target.floatAbi.empty())
    llc.push_back("-float-abi=" + target.floatAbi);
  if (!run(llvmBin + "/llc", llc, error))
    return failure();

  std::vector<std::string> cc = {"--target=" + target.triple, "-O2", "-fPIC",
                                 "-ffreestanding", "-fno-builtin-memcpy", "-c", rtC,
                                 "-o", rtObj};
  if (!target.cpu.empty())
    cc.push_back("-mcpu=" + target.cpu);
  if (!target.floatAbi.empty())
    cc.push_back("-mfloat-abi=" + target.floatAbi);
  if (!run(llvmBin + "/clang", cc, error))
    return failure();

  // A cross link has no sysroot, so libm and libgcc_s symbols stay undefined.
  // The runtime loads those libraries globally before this one
  // (src/runtime/share/loader.py).
  std::string library = dir + "/model.so";
  bool linked = target.native
                    ? run(llvmBin + "/clang", {"-shared", "-o", library, obj, rtObj, "-lm"}, error)
                    : run(llvmBin + "/ld.lld", {"-shared", "--hash-style=both", "-soname",
                                                "model.so", "-o", library, obj, rtObj},
                          error);
  if (!linked)
    return failure();
  if (!keepIr) {
    llvm::sys::fs::remove(ll);
    llvm::sys::fs::remove(rtC);
    llvm::sys::fs::remove(bc);
  }
  llvm::sys::fs::remove(canonical);
  llvm::sys::fs::remove(hinted);
  llvm::sys::fs::remove(obj);
  llvm::sys::fs::remove(rtObj);
  return success();
}

} // namespace backend
