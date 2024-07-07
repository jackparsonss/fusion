#include "ast/passes/pass.h"
#include "ast/ast.h"
#include "ast/passes/builtin.h"
#include "ast/passes/control_flow.h"
#include "ast/passes/def_ref.h"
#include "ast/passes/type_check.h"

constexpr bool debug = false;
#define try_visit(node, t, f)                                    \
    if (const shared_ptr<t> n = dynamic_pointer_cast<t>(node)) { \
        return f(n);                                             \
    }

void Pass::run_passes(std::shared_ptr<ast::Block> ast,
                      shared_ptr<SymbolTable> symtab) {
    std::vector<std::shared_ptr<Pass>> passes = {
        std::make_shared<ControlFlow>(),
        std::make_shared<DefRef>(symtab),
        std::make_shared<TypeCheck>(),
        std::make_shared<Builtin>(symtab),
    };

    for (std::shared_ptr<Pass>& pass : passes) {
        if (debug) {
            std::cout << "Running Pass: " << pass->name << std::endl;
        }
        pass->run(ast);
        if (debug) {
            std::cout << "Exiting Pass: " << pass->name << std::endl;
        }
    }
}

Pass::Pass(std::string name) {
    this->name = name;
}

void Pass::run(shared_ptr<ast::Block> ast) {
    visit_block(ast);
}

void Pass::visit(shared_ptr<ast::Node> node) {
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

    throw std::runtime_error("node not added to pass manager");
}

void Pass::visit_block(shared_ptr<ast::Block> node) {
    for (const auto& node : node->nodes) {
        visit(node);
    }
}

void Pass::visit_integer_literal(shared_ptr<ast::IntegerLiteral> node) {}
void Pass::visit_character_literal(shared_ptr<ast::CharacterLiteral> node) {}
void Pass::visit_boolean_literal(shared_ptr<ast::BooleanLiteral> node) {}
void Pass::visit_variable(shared_ptr<ast::Variable> node) {}
void Pass::visit_parameter(shared_ptr<ast::Parameter> node) {}
void Pass::visit_continue(shared_ptr<ast::Continue> node) {}
void Pass::visit_break(shared_ptr<ast::Break> node) {}

void Pass::visit_declaration(shared_ptr<ast::Declaration> node) {
    visit(node->var);
    visit(node->expr);
}

void Pass::visit_assignment(shared_ptr<ast::Assignment> node) {
    visit(node->var);
    visit(node->expr);
}

void Pass::visit_function(shared_ptr<ast::Function> node) {
    for (const auto& param : node->params) {
        visit(param);
    }

    visit(node->body);
}

void Pass::visit_call(shared_ptr<ast::Call> node) {
    for (const auto& arg : node->arguments) {
        visit(arg);
    }
}

void Pass::visit_return(shared_ptr<ast::Return> node) {
    visit(node->expr);
}

void Pass::visit_binary_operator(shared_ptr<ast::BinaryOperator> node) {
    visit(node->lhs);
    visit(node->rhs);
}

void Pass::visit_unary_operator(shared_ptr<ast::UnaryOperator> node) {
    visit(node->rhs);
}

void Pass::visit_conditional(shared_ptr<ast::Conditional> node) {
    visit(node->condition);
    visit(node->body);

    if (node->else_if.has_value()) {
        visit(node->else_if.value());
    }
}

void Pass::visit_loop(shared_ptr<ast::Loop> node) {
    visit(node->variable);
    visit(node->condition);
    visit(node->assignment);
    visit(node->body);
}
