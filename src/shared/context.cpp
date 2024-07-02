#include "shared/context.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
TypePtr ctx::i64;
TypePtr ctx::f32;
TypePtr ctx::none;
TypePtr ctx::bool_;

std::vector<TypePtr> ctx::primitives;
std::stack<mlir::LLVM::LLVMFuncOp> ctx::function_stack;

void ctx::initialize_context() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    builder = std::make_shared<mlir::OpBuilder>(&context);

    ctx::any = make_shared<Any>();
    ctx::i32 = make_shared<I32>();
    ctx::i64 = make_shared<I64>();
    ctx::f32 = make_shared<F32>();
    ctx::none = make_shared<None>();
    ctx::ch = make_shared<Character>();
    ctx::bool_ = make_shared<Boolean>();

    ctx::primitives = {
        ctx::ch, ctx::i32, ctx::i64, ctx::f32, ctx::bool_,
    };

    module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(builder->getUnknownLoc()));
}

mlir::LLVM::LLVMFuncOp ctx::current_function() {
    if (function_stack.empty()) {
        throw std::runtime_error("you are not within a function");
    }

    return function_stack.top();
}
