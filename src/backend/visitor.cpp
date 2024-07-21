#include <memory>

#include "ast/ast.h"
#include "backend/backend.h"
#include "backend/expressions/arithmetic.h"
#include "backend/expressions/flow.h"
#include "backend/types/boolean.h"
#include "backend/types/character.h"
#include "backend/types/integer.h"
#include "backend/utils.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    try_visit(node, ast::Conditional, this->visit_conditional);
    try_visit(node, ast::Loop, this->visit_loop);
    try_visit(node, ast::Continue, this->visit_continue);
    try_visit(node, ast::Break, this->visit_break);

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
    if (*node->get_type() == *ctx::i32) {
        return integer::create_i32(node->get_value());
    }

    return integer::create_i64(node->get_value());
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

    return utils::get_global(node->get_ref_name(), node->get_type());
    mlir::Value address = pair->second;
    return utils::load(address, node->get_type());
}

mlir::Value Backend::visit_declaration(shared_ptr<ast::Declaration> node) {
    std::string name = node->var->get_ref_name();
    mlir::Value expr = visit(node->expr);

    if (node->type == ast::DeclarationType::Local) {
        mlir::Value address = utils::stack_allocate(expr.getType());
        variables[name] = address;
        utils::store(address, expr);
        return nullptr;
    }

    if (node->type == ast::DeclarationType::Global) {
        utils::define_global(expr.getType(), name);
        mlir::LLVM::AddressOfOp address = utils::get_global_address(name);
        variables[name] = address;

        utils::store(address, expr);
        return nullptr;
    }

    throw std::runtime_error("Backend found declaration without a type");
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
    ctx::function_stack.push(func);

    for (size_t i = 0; i < node->params.size(); i++) {
        mlir::Value address = visit(node->params[i]);
        mlir::Value arg = func.getArgument(i);
        utils::store(address, arg);
    }

    visit(node->body);

    ctx::function_stack.pop();
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

mlir::Value Backend::visit_conditional(shared_ptr<ast::Conditional> node) {
    mlir::Value condition = visit(node->condition);
    mlir::Block* b_cond = ctx::current_function().addBlock();
    mlir::Block* b_else = ctx::current_function().addBlock();
    mlir::Block* b_exit = ctx::current_function().addBlock();

    flow::branch(condition, b_cond, b_else);

    ctx::builder->setInsertionPointToStart(b_cond);
    visit(node->body);
    flow::jump(b_exit);

    ctx::builder->setInsertionPointToStart(b_else);
    if (node->else_if.has_value()) {
        visit(node->else_if.value());
    }
    flow::jump(b_exit);

    ctx::builder->setInsertionPointToStart(b_exit);

    return nullptr;
}

mlir::Value Backend::visit_loop(shared_ptr<ast::Loop> node) {
    mlir::Block* b_cond = ctx::current_function().addBlock();
    mlir::Block* b_loop = ctx::current_function().addBlock();
    mlir::Block* b_assn = ctx::current_function().addBlock();
    mlir::Block* b_exit = ctx::current_function().addBlock();

    loop_conditions.push(b_assn);
    loop_exits.push(b_exit);

    visit(node->variable);

    flow::jump(b_cond);
    ctx::builder->setInsertionPointToStart(b_cond);

    mlir::Value condition = visit(node->condition);
    flow::branch(condition, b_loop, b_exit);

    ctx::builder->setInsertionPointToStart(b_loop);
    visit(node->body);
    flow::jump(b_assn);

    ctx::builder->setInsertionPointToStart(b_assn);
    visit(node->assignment);
    flow::jump(b_cond);

    ctx::builder->setInsertionPointToStart(b_exit);
    loop_conditions.pop();
    loop_exits.pop();

    return nullptr;
}

mlir::Value Backend::visit_continue(shared_ptr<ast::Continue> node) {
    if (loop_conditions.empty()) {
        throw std::runtime_error("backend found continue outside of a loop");
    }

    mlir::Block* b_body = ctx::current_function().addBlock();
    flow::jump(loop_conditions.top());

    ctx::builder->setInsertionPointToStart(b_body);

    return nullptr;
}

mlir::Value Backend::visit_break(shared_ptr<ast::Break> node) {
    if (loop_exits.empty()) {
        throw std::runtime_error("backend found break outside of a loop");
    }

    mlir::Block* b_body = ctx::current_function().addBlock();
    flow::jump(loop_exits.top());

    ctx::builder->setInsertionPointToStart(b_body);

    return nullptr;
}
