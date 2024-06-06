#include "backend/utils.h"

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
