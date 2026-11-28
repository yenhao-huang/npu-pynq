//===- vectorize_hints.h - loop vectorize hints on LLVM IR ------*- C++ -*-===//
#ifndef BACKEND_CPU_CODEGEN_VECTORIZE_HINTS_H
#define BACKEND_CPU_CODEGEN_VECTORIZE_HINTS_H

#include <string>

namespace backend {

// Read `in` (bitcode), mark every innermost loop that carries no
// floating-point value with llvm.loop.vectorize.enable, write `out`.
bool hintBitcode(const std::string &in, const std::string &out, std::string &error);

} // namespace backend

#endif // BACKEND_CPU_CODEGEN_VECTORIZE_HINTS_H
