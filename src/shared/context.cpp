#include "shared/context.h"
#include "shared/type/boolean.h"
#include "shared/type/character.h"
#include "shared/type/float.h"
#include "shared/type/integer.h"
#include "shared/type/type.h"

mlir::MLIRContext ctx::context;
std::unique_ptr<mlir::Location> ctx::loc =
    std::make_unique<mlir::Location>(mlir::UnknownLoc::get(&context));
std::shared_ptr<mlir::OpBuilder> ctx::builder;
std::unique_ptr<mlir::ModuleOp> ctx::module;

TypePtr ctx::ch;
TypePtr ctx::any;
TypePtr ctx::i32;
TypePtr ctx::f32;
TypePtr ctx::none;
TypePtr ctx::t_bool;

std::vector<TypePtr> ctx::primitives;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::any = make_shared<Any>();
    ctx::i32 = make_shared<I32>();
    ctx::f32 = make_shared<F32>();
    ctx::none = make_shared<None>();
    ctx::ch = make_shared<Character>();
    ctx::t_bool = make_shared<Boolean>();

    ctx::primitives = {
        ctx::ch,
        ctx::i32,
        ctx::f32,
        ctx::t_bool,
    };

    module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(builder->getUnknownLoc()));
}
