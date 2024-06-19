#include "backend/utils.h"
#include "backend/types/integer.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "shared/context.h"

mlir::Value utils::stack_allocate(mlir::Type type) {
    mlir::Value count = integer::create_i32(1);
    mlir::Value address = ctx::builder->create<mlir::LLVM::AllocaOp>(
        *ctx::loc, make_pointer(type), count);

    return address;
}

mlir::Type utils::make_pointer(mlir::Type type) {
    return mlir::LLVM::LLVMPointerType::get(type);
}

mlir::Value utils::call(mlir::LLVM::LLVMFuncOp func, mlir::ValueRange params) {
    mlir::LLVM::CallOp op =
        ctx::builder->create<mlir::LLVM::CallOp>(*ctx::loc, func, params);

    return op.getResult();
}

mlir::Value utils::load(mlir::Value address) {
    mlir::Value value =
        ctx::builder->create<mlir::LLVM::LoadOp>(*ctx::loc, address);

    return value;
}

void utils::store(mlir::Value address, mlir::Value value) {
    ctx::builder->create<mlir::LLVM::StoreOp>(*ctx::loc, value, address);
}

mlir::LLVM::LLVMFuncOp utils::get_function(shared_ptr<ast::Function> func) {
    mlir::Type return_type = func->get_type()->get_mlir();

    std::vector<mlir::Type> param_types(func->params.size());
    for (size_t i = 0; i < func->params.size(); i++) {
        param_types[i] = func->params[i]->var->get_type()->get_mlir();
    }
    mlir::ArrayRef<mlir::Type> params(param_types);

    return mlir::LLVM::lookupOrCreateFn(*ctx::module, func->get_ref_name(),
                                        params, return_type);
}
