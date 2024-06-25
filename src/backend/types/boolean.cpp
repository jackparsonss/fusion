#include "backend/types/boolean.h"
#include "shared/context.h"

mlir::Value boolean::create_bool(bool value) {
    mlir::Value val = ctx::builder->create<mlir::LLVM::ConstantOp>(
        *ctx::loc, ctx::t_bool->get_mlir(), value);

    return val;
}
