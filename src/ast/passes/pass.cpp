#include "ast/passes/pass.h"
#include "ast/ast.h"
#include "ast/passes/def_ref.h"

constexpr bool debug = false;
#define try_visit(node, t, f)                                    \
    if (const shared_ptr<t> n = dynamic_pointer_cast<t>(node)) { \
        return f(n);                                             \
    }

void pass::run_passes(std::shared_ptr<ast::Block> ast,
                      shared_ptr<SymbolTable> symtab) {
    std::vector<std::shared_ptr<Pass>> passes = {
        std::make_shared<DefRef>(symtab),
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
    try_visit(node, ast::Variable, this->visit_variable);
    try_visit(node, ast::Declaration, this->visit_declaration);
    try_visit(node, ast::Function, this->visit_function);
    try_visit(node, ast::Call, this->visit_call);
    try_visit(node, ast::Parameter, this->visit_parameter);
}

void Pass::visit_block(shared_ptr<ast::Block> node) {
    for (const auto& node : node->nodes) {
        visit(node);
    }
}

void Pass::visit_integer_literal(shared_ptr<ast::IntegerLiteral> node) {}

void Pass::visit_declaration(shared_ptr<ast::Declaration> node) {
    visit(node->var);
    visit(node->expr);
}

void Pass::visit_variable(shared_ptr<ast::Variable> node) {}

void Pass::visit_parameter(shared_ptr<ast::Parameter> node) {}

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
