#pragma once

#include "ast/ast.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace utils {
mlir::Value stack_allocate(mlir::Type type);
mlir::Type make_pointer(mlir::Type type);
mlir::Value call(mlir::LLVM::LLVMFuncOp func, mlir::ValueRange params = {});
mlir::Value load(mlir::Value address);
mlir::LLVM::LLVMFuncOp get_function(shared_ptr<ast::Function> func);
mlir::Value get_global(std::string name);
void define_global(mlir::Type, std::string name);
void store(mlir::Value address, mlir::Value value);
}  // namespace utils
