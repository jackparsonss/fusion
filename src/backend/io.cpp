#include "backend/io.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "shared/context.h"

void io::print(mlir::Value value) {
    mlir::LLVM::LLVMFuncOp func = mlir::LLVM::lookupOrCreateFn(
        *ctx::module, "print_integer", ctx::i32->get_mlir(),
        ctx::none->get_mlir());

    ctx::builder->create<mlir::LLVM::CallOp>(*ctx::loc, func, value);
}
