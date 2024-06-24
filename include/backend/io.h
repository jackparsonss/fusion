#pragma once

#include "mlir/IR/Value.h"
#include "shared/type.h"

namespace io {
void printf(mlir::Value value, TypePtr type);
}  // namespace io
