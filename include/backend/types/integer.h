#pragma once

#include "mlir/IR/Value.h"

namespace integer {
mlir::Value create_i32(int value);
mlir::Value create_i64(long long value);
}  // namespace integer
