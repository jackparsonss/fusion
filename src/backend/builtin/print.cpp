#include "backend/builtin/print.h"
#include "backend/io.h"
#include "shared/context.h"

void builtin::define_all() {
    define_print(ctx::i32);
}

void builtin::define_print(std::shared_ptr<Type> type) {
    // auto ty = mlir::LLVM::LLVMFunctionType::get(ctx::none->get_mlir(),
    //                                             {type->get_mlir()}, false);
    // mlir::LLVM::LLVMFuncOp func =
    //     ctx::builder->create<mlir::LLVM::LLVMFuncOp>(*ctx::loc, "print", ty);
    //
    // mlir::Block* body = func.addEntryBlock();
    // ctx::builder->setInsertionPointToStart(body);
    //
    // mlir::Value arg = func.getArgument(0);
    // io::print(arg);
    //
    // ctx::builder->create<mlir::LLVM::ReturnOp>(*ctx::loc, mlir::Value{});
    // ctx::builder->setInsertionPointToEnd(ctx::module->getBody());
}
