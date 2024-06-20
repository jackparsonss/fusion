#include "backend/types/character.h"
#include "shared/context.h"

mlir::Value character::create_ch(char value) {
    mlir::Value val = ctx::builder->create<mlir::LLVM::ConstantOp>(
        *ctx::loc, ctx::ch->get_mlir(), value);

    return val;
}
