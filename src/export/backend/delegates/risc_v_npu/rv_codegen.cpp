//===- rv_codegen.cpp - LLVM dialect -> flat RV32IMC binary ---------------===//
//
//   LLVM dialect  --mlir::translateModuleToLLVMIR-->  LLVM IR
//     + crt0 as module-level assembly (assembled by LLVM MC with the code)
//     --O2 (new pass manager), RISCV TargetMachine-->  one ELF object
//     --check: no absolute relocation in a loaded section-->
//     --<llvm-bin>/ld.lld -T link.ld-->  ELF linked at 0
//     --flatten PT_LOAD segments-->  bytes + memory footprint
//
// Codegen runs in process, so this pynqo-backend must be built against an
// LLVM whose LLVM_TARGETS_TO_BUILD has RISCV (CMake defines
// PYNQO_HAVE_RISCV). Linking uses the ld.lld executable, as the CPU path
// (cpu/toolchain.cpp) already does: the LLVM 22 packages this repository
// builds against ship ld.lld but not lld's libraries. A single object plus a
// linker script needs nothing lld-specific, so an in-process linker would
// only add a dependency.
//
// Position independence: the code model is medany (every symbol reference
// is auipc-relative) with the static relocation model (no GOT), and jump
// tables are disabled (their entries would be absolute addresses). The
// object check below enforces the result, so a binary that would only work
// at address 0 is never written.
//
//===----------------------------------------------------------------------===//

#include "rv_codegen.h"

#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <memory>
#include <mutex>

#ifdef PYNQO_HAVE_RISCV
extern "C" {
void LLVMInitializeRISCVTargetInfo();
void LLVMInitializeRISCVTarget();
void LLVMInitializeRISCVTargetMC();
void LLVMInitializeRISCVAsmPrinter();
void LLVMInitializeRISCVAsmParser();
}
#endif

using namespace mlir;

namespace backend {

#ifndef PYNQO_HAVE_RISCV

bool rvCodegenAvailable() { return false; }

LogicalResult rvCodegen(ModuleOp, const RvTarget &, llvm::StringRef, llvm::StringRef,
                        const std::string &, const std::string &, const std::string &,
                        bool, RvImage &, std::string &error) {
  error = "risc_v_npu: this pynqo-backend was built against an LLVM without the RISCV "
          "target (LLVM_TARGETS_TO_BUILD)";
  return failure();
}

#else

namespace {

void initRiscv() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmPrinter();
    LLVMInitializeRISCVAsmParser();
  });
}

bool run(const std::string &program, const std::vector<std::string> &args,
         std::string &error) {
  std::vector<llvm::StringRef> argv{program};
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

bool writeFile(const std::string &path, llvm::StringRef data, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  os << data;
  return true;
}

// Relocations that resolve to an absolute address: wrong once the image is
// loaded anywhere but 0.
bool isAbsolute(uint64_t type) {
  using namespace llvm::ELF;
  switch (type) {
  case R_RISCV_32:
  case R_RISCV_64:
  case R_RISCV_HI20:
  case R_RISCV_LO12_I:
  case R_RISCV_LO12_S:
  case R_RISCV_GOT_HI20:
  case R_RISCV_RELATIVE:
  case R_RISCV_COPY:
  case R_RISCV_JUMP_SLOT:
    return true;
  default:
    return false;
  }
}

bool checkRelocations(llvm::MemoryBufferRef object, std::string &error) {
  auto file = llvm::object::ObjectFile::createObjectFile(object);
  if (!file) {
    error = "risc_v_npu: cannot read the object: " + llvm::toString(file.takeError());
    return false;
  }
  for (const llvm::object::SectionRef &section : (*file)->sections()) {
    auto target = section.getRelocatedSection();
    if (!target) {
      llvm::consumeError(target.takeError());
      continue;
    }
    if (*target == (*file)->section_end() ||
        !(llvm::object::ELFSectionRef(**target).getFlags() & llvm::ELF::SHF_ALLOC))
      continue;
    for (const llvm::object::RelocationRef &rel : section.relocations())
      if (isAbsolute(rel.getType())) {
        llvm::SmallString<32> name;
        rel.getTypeName(name);
        error = "risc_v_npu: the program is not position independent (" + name.str().str() +
                " relocation)";
        return false;
      }
  }
  return true;
}

// The PT_LOAD segments of a linked ELF as one image from address 0.
bool flatten(llvm::MemoryBufferRef elf, RvImage &image, std::string &error) {
  auto file = llvm::object::ELF32LEObjectFile::create(elf);
  if (!file) {
    error = "risc_v_npu: cannot read the linked program: " + llvm::toString(file.takeError());
    return false;
  }
  const auto &ef = file->getELFFile();
  if (ef.getHeader().e_entry != 0) {
    error = "risc_v_npu: the program's entry is not at offset 0";
    return false;
  }
  auto headers = ef.program_headers();
  if (!headers) {
    error = "risc_v_npu: " + llvm::toString(headers.takeError());
    return false;
  }
  uint64_t fileEnd = 0, memEnd = 0;
  for (const auto &ph : *headers)
    if (ph.p_type == llvm::ELF::PT_LOAD) {
      if (ph.p_filesz)
        fileEnd = std::max<uint64_t>(fileEnd, ph.p_vaddr + ph.p_filesz);
      memEnd = std::max<uint64_t>(memEnd, ph.p_vaddr + ph.p_memsz);
    }
  if (fileEnd == 0 || memEnd > (1u << 24)) {
    error = "risc_v_npu: the linked program has an unexpected layout";
    return false;
  }
  image.bytes.assign(fileEnd, 0);
  llvm::StringRef data = elf.getBuffer();
  for (const auto &ph : *headers)
    if (ph.p_type == llvm::ELF::PT_LOAD && ph.p_filesz) {
      if (ph.p_offset + ph.p_filesz > data.size()) {
        error = "risc_v_npu: truncated program";
        return false;
      }
      std::copy_n(data.data() + ph.p_offset, ph.p_filesz, image.bytes.begin() + ph.p_vaddr);
    }
  image.memBytes = static_cast<int64_t>(memEnd);
  return true;
}

} // namespace

bool rvCodegenAvailable() { return true; }

LogicalResult rvCodegen(ModuleOp module, const RvTarget &target, llvm::StringRef runtimeAsm,
                        llvm::StringRef linkScript, const std::string &workDir,
                        const std::string &name, const std::string &llvmBin, bool keep,
                        RvImage &image, std::string &error) {
  initRiscv();
  llvm::Triple triple(target.triple);
  std::string lookupError;
  const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(triple, lookupError);
  if (!llvmTarget) {
    error = "risc_v_npu: " + lookupError;
    return failure();
  }
  llvm::TargetOptions options;
  options.MCOptions.ABIName = target.abi;
  options.FunctionSections = false;
  std::unique_ptr<llvm::TargetMachine> machine(llvmTarget->createTargetMachine(
      triple, target.cpu, target.features, options, llvm::Reloc::Static,
      llvm::CodeModel::Medium, llvm::CodeGenOptLevel::Default));
  if (!machine) {
    error = "risc_v_npu: cannot create the RISC-V target machine";
    return failure();
  }

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> ir = translateModuleToLLVMIR(module, context, name);
  if (!ir) {
    error = "risc_v_npu: translation to LLVM IR failed";
    return failure();
  }
  ir->setTargetTriple(triple);
  ir->setDataLayout(machine->createDataLayout());
  ir->addModuleFlag(llvm::Module::Error, "target-abi",
                    llvm::MDString::get(context, target.abi));
  for (llvm::Function &fn : *ir) {
    if (fn.isDeclaration())
      continue;
    fn.addFnAttr("target-cpu", target.cpu);
    fn.addFnAttr("target-features", target.features);
    fn.addFnAttr("no-jump-tables", "true");
    fn.addFnAttr(llvm::Attribute::NoUnwind);
  }
  ir->appendModuleInlineAsm(runtimeAsm);

  {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PassBuilder pb(machine.get());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2).run(*ir, mam);
  }

  llvm::sys::fs::create_directories(workDir);
  std::string base = workDir + "/" + name;
  if (keep) {
    std::error_code ec;
    llvm::raw_fd_ostream os(base + ".ll", ec, llvm::sys::fs::OF_Text);
    if (!ec)
      ir->print(os, nullptr);
  }

  llvm::SmallString<0> object;
  {
    llvm::raw_svector_ostream os(object);
    llvm::legacy::PassManager pm;
    if (machine->addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::ObjectFile)) {
      error = "risc_v_npu: the RISC-V target cannot emit an object";
      return failure();
    }
    pm.run(*ir);
  }
  if (!checkRelocations(llvm::MemoryBufferRef(object.str(), name + ".o"), error))
    return failure();

  std::string obj = base + ".o", script = base + ".ld", elf = base + ".elf";
  if (!writeFile(obj, object.str(), error) || !writeFile(script, linkScript, error))
    return failure();
  if (!run(llvmBin + "/ld.lld", {"-nostdlib", "-static", "-T", script, "-o", elf, obj}, error))
    return failure();
  auto linked = llvm::MemoryBuffer::getFile(elf, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!linked) {
    error = "risc_v_npu: cannot read " + elf;
    return failure();
  }
  if (!flatten((*linked)->getMemBufferRef(), image, error))
    return failure();
  if (!keep) {
    llvm::sys::fs::remove(obj);
    llvm::sys::fs::remove(script);
    llvm::sys::fs::remove(elf);
  }
  return success();
}

#endif // PYNQO_HAVE_RISCV

} // namespace backend
