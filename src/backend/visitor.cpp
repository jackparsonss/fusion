#include <memory>

#include "ast/ast.h"
#include "backend/backend.h"
#include "backend/expressions/arithmetic.h"
#include "backend/types/boolean.h"
#include "backend/types/character.h"
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

mlir::Value Backend::visit(shared_ptr<ast::Node> node) {
    try_visit(node, ast::Block, this->visit_block);
    try_visit(node, ast::IntegerLiteral, this->visit_integer_literal);
    try_visit(node, ast::CharacterLiteral, this->visit_character_literal);
    try_visit(node, ast::BooleanLiteral, this->visit_boolean_literal);
    try_visit(node, ast::Variable, this->visit_variable);
    try_visit(node, ast::Declaration, this->visit_declaration);
    try_visit(node, ast::Assignment, this->visit_assignment);
    try_visit(node, ast::Function, this->visit_function);
    try_visit(node, ast::Call, this->visit_call);
    try_visit(node, ast::Parameter, this->visit_parameter);
    try_visit(node, ast::Return, this->visit_return);
    try_visit(node, ast::BinaryOperator, this->visit_binary_operator);
    try_visit(node, ast::UnaryOperator, this->visit_unary_operator);

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

mlir::Value Backend::visit_character_literal(
    shared_ptr<ast::CharacterLiteral> node) {
    return character::create_ch(node->get_value());
}

mlir::Value Backend::visit_boolean_literal(
    shared_ptr<ast::BooleanLiteral> node) {
    return boolean::create_bool(node->get_value());
}

mlir::Value Backend::visit_variable(shared_ptr<ast::Variable> node) {
    auto pair = variables.find(node->get_ref_name());
    if (pair == variables.end()) {
        throw std::runtime_error("backend found undefined variable: " +
                                 node->get_name());
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

mlir::Value Backend::visit_assignment(shared_ptr<ast::Assignment> node) {
    std::string name = node->var->get_ref_name();
    mlir::Value expr = visit(node->expr);
    mlir::Value address = variables[name];

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

mlir::Value Backend::visit_binary_operator(
    shared_ptr<ast::BinaryOperator> node) {
    mlir::Value lhs = visit(node->lhs);
    mlir::Value rhs = visit(node->rhs);

    return arithmetic::binary_operation(lhs, rhs, node->type, node->get_type());
}

mlir::Value Backend::visit_unary_operator(shared_ptr<ast::UnaryOperator> node) {
    mlir::Value rhs = visit(node->rhs);

    return arithmetic::unary_operation(rhs, node->type, node->get_type());
}
