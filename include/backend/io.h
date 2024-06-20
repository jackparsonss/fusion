#pragma once

#include "mlir/IR/Value.h"
#include "shared/type.h"

namespace io {
void print(mlir::Value, std::shared_ptr<Type>);
}  // namespace io
