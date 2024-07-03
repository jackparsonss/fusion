#pragma once

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

namespace flow {
void branch(mlir::Value condition, mlir::Block* b_true, mlir::Block* b_false);
void jump(mlir::Block* block);
}  // namespace flow
