#pragma once

#include "ast/ast.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace utils {
mlir::Value stack_allocate(mlir::Type type);
mlir::Type make_pointer(mlir::Type type);
mlir::Value call(mlir::LLVM::LLVMFuncOp func, mlir::ValueRange params = {});
mlir::Value load(mlir::Value address);
mlir::LLVM::LLVMFuncOp get_function(shared_ptr<ast::Function> func);
void store(mlir::Value address, mlir::Value value);
}  // namespace utils
