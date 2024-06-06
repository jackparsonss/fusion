#include "backend/types/integer.h"

mlir::Value integer::create_i32(int value) {
    mlir::Value val = ctx::builder->create<mlir::LLVM::ConstantOp>(
        *ctx::loc, ctx::i32, value);

    return val;
}
