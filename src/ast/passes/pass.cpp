#include "ast/passes/pass.h"
#include "ast/passes/def_ref.h"

constexpr bool debug = false;

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
    if (const auto block = dynamic_pointer_cast<ast::Block>(node)) {
        visit_block(block);
    }

    if (const auto literal = dynamic_pointer_cast<ast::IntegerLiteral>(node)) {
        visit_integer_literal(literal);
    }

    if (const auto decl = dynamic_pointer_cast<ast::Declaration>(node)) {
        visit_declaration(decl);
    }

    if (const auto var = dynamic_pointer_cast<ast::Variable>(node)) {
        visit_variable(var);
    }

    if (const auto func = dynamic_pointer_cast<ast::Function>(node)) {
        visit_function(func);
    }

    if (const auto call = dynamic_pointer_cast<ast::Call>(node)) {
        visit_call(call);
    }
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

void Pass::visit_function(shared_ptr<ast::Function> node) {
    visit(node->body);
}

void Pass::visit_call(shared_ptr<ast::Call> node) {
    for (const auto& arg : node->arguments) {
        visit(arg);
    }
}
