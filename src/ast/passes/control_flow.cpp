#include "ast/passes/control_flow.h"
#include "errors/errors.h"

ControlFlow::ControlFlow() : Pass("Control Flow") {
    this->in_function = false;
    this->in_loop = false;
}

void ControlFlow::visit_function(shared_ptr<ast::Function> node) {
    for (const auto& param : node->params) {
        visit(param);
    }

    in_function = true;
    visit(node->body);
    in_function = false;
}

void ControlFlow::visit_return(shared_ptr<ast::Return> node) {
    if (!in_function) {
        throw SyntaxError(node->token->getLine(),
                          "found `return` outside of function");
    }
    visit(node->expr);
}

void ControlFlow::visit_loop(shared_ptr<ast::Loop> node) {
    in_loop = true;
    visit(node->variable);
    visit(node->condition);
    visit(node->assignment);
    visit(node->body);
    in_loop = false;
}

void ControlFlow::visit_continue(shared_ptr<ast::Continue> node) {
    if (!in_loop) {
        throw SyntaxError(node->token->getLine(),
                          "found `continue` outside of loop");
    }
}

void ControlFlow::visit_break(shared_ptr<ast::Break> node) {
    if (!in_loop) {
        throw SyntaxError(node->token->getLine(),
                          "found `break` outside of loop");
    }
}
