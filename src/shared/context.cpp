#include "shared/context.h"
#include "backend/builtin/print.h"

mlir::MLIRContext ctx::context;
std::unique_ptr<mlir::Location> ctx::loc =
    std::make_unique<mlir::Location>(mlir::UnknownLoc::get(&context));
std::shared_ptr<mlir::OpBuilder> ctx::builder;
std::unique_ptr<mlir::ModuleOp> ctx::module;

shared_ptr<Type> ctx::ch;
shared_ptr<Type> ctx::i32;
shared_ptr<Type> ctx::none;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::ch = std::make_shared<Type>(Type::ch);
    ctx::i32 = std::make_shared<Type>(Type::i32);
    ctx::none = std::make_shared<Type>(Type::none);
    module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(builder->getUnknownLoc()));

    builtin::define_all();
}
