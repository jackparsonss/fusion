#pragma once

#include "backend/context.h"

#include "mlir/IR/Value.h"

namespace integer {
mlir::Value create_i32(int value);
}  // namespace integer
