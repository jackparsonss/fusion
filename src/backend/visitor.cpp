#include <memory>
#include "ast/ast.h"
#include "backend/backend.h"
#include "backend/types/integer.h"
#include "backend/utils.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/IR/ValueRange.h"
#include "shared/context.h"

#define to_node(a, b) dynamic_pointer_cast<b>(a)
#define try_visit(node, t, f)                       \
    if (const shared_ptr<t> n = to_node(node, t)) { \
        return f(n);                                \
    }

BackendVisitor::BackendVisitor(shared_ptr<ast::Block> ast) {
    this->ast = ast;
}

mlir::Value BackendVisitor::visit(shared_ptr<ast::Node> node) {
    try_visit(node, ast::Block, this->visit_block);
    try_visit(node, ast::IntegerLiteral, this->visit_integer_literal);
    try_visit(node, ast::Variable, this->visit_variable);
    try_visit(node, ast::Declaration, this->visit_declaration);
    try_visit(node, ast::Function, this->visit_function);
    try_visit(node, ast::Call, this->visit_call);
    try_visit(node, ast::Parameter, this->visit_parameter);
    try_visit(node, ast::Return, this->visit_return);

    throw std::runtime_error("node not added to backend visit function");
}

mlir::Value Backend::visit_block(shared_ptr<ast::Block> node) {
    for (shared_ptr<ast::Node> const& statement : node->nodes) {
        visit(statement);
    }

    return nullptr;
}

mlir::Value Backend::visit_integer_literal(
    shared_ptr<ast::IntegerLiteral> node) {
    return integer::create_i32(node->get_value());
}

mlir::Value Backend::visit_variable(shared_ptr<ast::Variable> node) {
    auto pair = variables.find(node->get_ref_name());
    if (pair == variables.end()) {
        throw std::runtime_error("backend found undefined variable: " +
                                 node->get_ref_name());
    }

    mlir::Value address = pair->second;
    return utils::load(address);
}

mlir::Value Backend::visit_declaration(shared_ptr<ast::Declaration> node) {
    std::string name = node->var->get_ref_name();
    mlir::Value expr = visit(node->expr);
    mlir::Value address = utils::stack_allocate(expr.getType());

    variables[name] = address;
    utils::store(address, expr);

    return nullptr;
}

mlir::Value Backend::visit_parameter(shared_ptr<ast::Parameter> node) {
    std::string name = node->var->get_ref_name();
    mlir::Value address =
        utils::stack_allocate(node->var->get_type()->get_mlir());

    variables[name] = address;

    return address;
}

mlir::Value Backend::visit_function(shared_ptr<ast::Function> node) {
    mlir::LLVM::LLVMFuncOp func = utils::get_function(node);

    mlir::Block* b_body = func.addEntryBlock();
    ctx::builder->setInsertionPointToStart(b_body);

    for (size_t i = 0; i < node->params.size(); i++) {
        mlir::Value address = visit(node->params[i]);
        mlir::Value arg = func.getArgument(i);
        utils::store(address, arg);
    }

    visit(node->body);

    ctx::builder->setInsertionPointToEnd(ctx::module->getBody());

    return nullptr;
}

mlir::Value Backend::visit_call(shared_ptr<ast::Call> node) {
    mlir::LLVM::LLVMFuncOp func = utils::get_function(node->get_function());
    std::vector<mlir::Value> args(node->arguments.size());

    for (size_t i = 0; i < node->arguments.size(); i++) {
        mlir::Value arg = visit(node->arguments[i]);
        args[i] = arg;
    }

    return utils::call(func, mlir::ValueRange(args));
}

mlir::Value Backend::visit_return(shared_ptr<ast::Return> node) {
    mlir::Value expr = visit(node->expr);
    ctx::builder->create<mlir::LLVM::ReturnOp>(*ctx::loc, expr);

    return nullptr;
}
