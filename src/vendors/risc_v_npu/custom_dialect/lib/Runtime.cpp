//===- Runtime.cpp - the CVA6 runtime linked into every program -----------===//
//
// runtime/crt0.S and runtime/link.ld, embedded by CMake (RvNpuRuntime.inc),
// and the `.equ` block crt0.S is assembled after: the RvNpuMmio.h constants
// it uses, so the header stays the single definition of the MMIO window.
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuMmio.h"
#include "RvNpu/RvNpuPasses.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace {
#include "RvNpuRuntime.inc"
} // namespace

llvm::StringRef mlir::rvnpu::crt0Source() { return kCrt0Source; }

llvm::StringRef mlir::rvnpu::linkScript() { return kLinkScript; }

std::string mlir::rvnpu::mmioEquates() {
  std::string text;
  llvm::raw_string_ostream os(text);
  auto equ = [&](const char *name, unsigned long long value) {
    os << "    .equ " << name << ", " << llvm::format_hex(value, 10) << "\n";
  };
#define RVNPU_EQU(name) equ(#name, static_cast<unsigned long long>(name))
  RVNPU_EQU(RVNPU_MMIO_BASE);
  RVNPU_EQU(RVNPU_MMIO_DONE);
  RVNPU_EQU(RVNPU_DONE_SUCCESS);
  RVNPU_EQU(RVNPU_DONE_TRAP);
#undef RVNPU_EQU
  return text;
}
