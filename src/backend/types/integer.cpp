#include "backend/types/integer.h"
#include "shared/context.h"

mlir::Value integer::create_i32(int value) {
    mlir::Value val = ctx::builder->create<mlir::LLVM::ConstantOp>(
        *ctx::loc, ctx::i32->get_mlir(), value);

    return val;
}

mlir::Value integer::create_i64(long long value) {
    mlir::Value val = ctx::builder->create<mlir::LLVM::ConstantOp>(
        *ctx::loc, ctx::i64->get_mlir(), value);

    return val;
}
