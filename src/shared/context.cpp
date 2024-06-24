#include "shared/context.h"

mlir::MLIRContext ctx::context;
std::unique_ptr<mlir::Location> ctx::loc =
    std::make_unique<mlir::Location>(mlir::UnknownLoc::get(&context));
std::shared_ptr<mlir::OpBuilder> ctx::builder;
std::unique_ptr<mlir::ModuleOp> ctx::module;

TypePtr ctx::ch;
TypePtr ctx::i32;
TypePtr ctx::f32;
TypePtr ctx::none;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::ch = std::make_shared<Type>(Type::ch);
    ctx::i32 = std::make_shared<Type>(Type::i32);
    ctx::f32 = std::make_shared<Type>(Type::f32);
    ctx::none = std::make_shared<Type>(Type::none);
    module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(builder->getUnknownLoc()));
}
