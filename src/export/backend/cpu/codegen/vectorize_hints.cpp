//===- vectorize_hints.cpp - loop vectorize hints on LLVM IR --------------===//
//
// Runs between the two opt runs in toolchain.cpp, on bitcode.
//
//===----------------------------------------------------------------------===//

#include "cpu/codegen/vectorize_hints.h"

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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace backend {
namespace {

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
// interchangeForUnitStride (cpu/lowering/interchange.cpp) has already turned
// the hot reductions into loops over an output dimension, which carry nothing.
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

} // namespace

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

} // namespace backend
