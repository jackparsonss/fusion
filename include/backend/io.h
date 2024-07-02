#pragma once

#include "mlir/IR/Value.h"
#include "shared/type/type.h"

namespace io {
void printf(mlir::Value value, TypePtr type);
void println(mlir::Value value, TypePtr type);
}  // namespace io
