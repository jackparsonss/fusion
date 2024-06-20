#include "backend/io.h"
#include <string>

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "shared/context.h"

void io::print(mlir::Value value, std::shared_ptr<Type> type) {
    std::string name = "print_" + type->get_name();
    mlir::LLVM::LLVMFuncOp func = mlir::LLVM::lookupOrCreateFn(
        *ctx::module, name, type->get_mlir(), ctx::none->get_mlir());

    ctx::builder->create<mlir::LLVM::CallOp>(*ctx::loc, func, value);
}
