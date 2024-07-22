#include "backend/utils.h"
#include "backend/types/integer.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "shared/context.h"

mlir::Value utils::stack_allocate(mlir::Type type) {
    mlir::Value count = integer::create_i32(1);
    mlir::Value address = ctx::builder->create<mlir::LLVM::AllocaOp>(
        *ctx::loc, make_pointer(), type, count);

    return address;
}

mlir::Type utils::make_pointer() {
    return mlir::LLVM::LLVMPointerType::get(&ctx::context);
}

mlir::Value utils::call(mlir::LLVM::LLVMFuncOp func, mlir::ValueRange params) {
    mlir::LLVM::CallOp op =
        ctx::builder->create<mlir::LLVM::CallOp>(*ctx::loc, func, params);

    return op.getResult();
}

mlir::Value utils::load(mlir::Value address, TypePtr type) {
    mlir::Value value = ctx::builder->create<mlir::LLVM::LoadOp>(
        *ctx::loc, type->get_mlir(), address);

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

void utils::define_global(mlir::Type type, std::string name) {
    ctx::builder->create<mlir::LLVM::GlobalOp>(
        *ctx::loc, type, false, mlir::LLVM::Linkage::Internal, name, nullptr);
}

mlir::LLVM::AddressOfOp utils::get_global_address(std::string name) {
    mlir::LLVM::GlobalOp global;
    if (!(global = ctx::module->lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
        throw std::runtime_error("backend failed to get global variable");
    }

    return ctx::builder->create<mlir::LLVM::AddressOfOp>(*ctx::loc, global);
}

mlir::Value utils::get_global(std::string name, TypePtr type) {
    return load(get_global_address(name), type);
}
