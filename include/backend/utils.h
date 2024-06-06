#pragma once

#include "backend/context.h"
#include "backend/types/integer.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace utils {
mlir::Value stack_allocate(mlir::Type type);
mlir::Type make_pointer(mlir::Type type);
mlir::Value call(mlir::LLVM::LLVMFuncOp func, mlir::ValueRange params = {});
mlir::Value load(mlir::Value address);
void store(mlir::Value address, mlir::Value value);
}  // namespace utils
