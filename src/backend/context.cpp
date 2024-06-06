#include "backend/context.h"

mlir::MLIRContext ctx::context;
std::unique_ptr<mlir::Location> ctx::loc =
    std::make_unique<mlir::Location>(mlir::UnknownLoc::get(&context));
std::shared_ptr<mlir::OpBuilder> ctx::builder;
std::unique_ptr<mlir::ModuleOp> ctx::module;

mlir::Type ctx::t_int;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::t_int = builder->getI32Type();
    module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(builder->getUnknownLoc()));
}
