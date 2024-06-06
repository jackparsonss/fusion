#include "backend/io.h"

void io::print(mlir::Value value) {
    mlir::LLVM::LLVMFuncOp func = mlir::LLVM::lookupOrCreateFn(
        *ctx::module, "print_integer", ctx::i32, ctx::t_void);

    ctx::builder->create<mlir::LLVM::CallOp>(*ctx::loc, func, value);
}
