#pragma once

#include "backend/context.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace io {
void print(mlir::Value);
}  // namespace io
