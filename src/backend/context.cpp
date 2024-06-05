#include "backend/context.h"

mlir::MLIRContext ctx::context;
mlir::Location ctx::loc = mlir::UnknownLoc::get(&context);
std::shared_ptr<mlir::OpBuilder> ctx::builder;
mlir::Type ctx::t_int;
mlir::ModuleOp ctx::module;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::t_int = builder->getI32Type();
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
}
