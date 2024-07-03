#include "ast/passes/type_check.h"
#include "ast/ast.h"
#include "errors/errors.h"
#include "shared/context.h"

TypeCheck::TypeCheck() : Pass("Typecheck") {}

void TypeCheck::visit_declaration(shared_ptr<ast::Declaration> node) {
    visit(node->var);
    visit(node->expr);

    TypePtr var = node->var->get_type();
    TypePtr expr = node->expr->get_type();
    if (*var == *ctx::i64 && *expr == *ctx::i32) {
        node->expr->set_type(ctx::i64);
        return;
    }

    if (*var != *expr) {
        throw TypeError(node->token->getLine(),
                        "mismatched lhs(" + var->get_name() + ") and rhs(" +
                            expr->get_name() + ") types on declaration");
    }
}

void TypeCheck::visit_assignment(shared_ptr<ast::Assignment> node) {
    visit(node->var);
    visit(node->expr);

    Type var = *node->var->get_type();
    Type expr = *node->expr->get_type();
    if (var != expr) {
        throw TypeError(node->token->getLine(),
                        "mismatched lhs(" + var.get_name() + ") and rhs(" +
                            expr.get_name() + ") types on assignment");
    }
}

void TypeCheck::visit_function(shared_ptr<ast::Function> node) {
    for (const auto& param : node->params) {
        visit(param);
    }

    func_stack.push(node);
    visit(node->body);
    func_stack.pop();
}

void TypeCheck::visit_call(shared_ptr<ast::Call> node) {
    size_t line = node->token->getLine();
    if (node->arguments.size() != node->get_function()->params.size()) {
        throw TypeError(line, "calling function " + node->get_name() +
                                  " with the wrong number of parameters");
    }

    for (size_t i = 0; i < node->arguments.size(); i++) {
        visit(node->arguments[i]);

        Type arg = *node->arguments[i]->get_type();
        Type param = *node->get_function()->params[i]->var->get_type();
        if (arg != param) {
            throw TypeError(line, "mismatched argument(" + arg.get_name() +
                                      ") and parameter(" + param.get_name() +
                                      ") types on function call");
        }
    }
}

void TypeCheck::visit_return(shared_ptr<ast::Return> node) {
    visit(node->expr);

    Type func = *func_stack.top()->get_type();
    Type expr = *node->expr->get_type();
    if (expr != func) {
        throw TypeError(node->token->getLine(),
                        "returned value type(" + expr.get_name() +
                            ") does not match function type(" +
                            func.get_name() + ")");
    }
}

void TypeCheck::visit_binary_operator(shared_ptr<ast::BinaryOperator> node) {
    visit(node->lhs);
    visit(node->rhs);

    size_t line = node->token->getLine();
    TypePtr lhs = node->lhs->get_type();
    TypePtr rhs = node->rhs->get_type();

    if (*lhs != *rhs) {
        throw TypeError(line, "mismatched lhs(" + lhs->get_name() +
                                  ") and rhs(" + rhs->get_name() +
                                  ") types on binary operator: " +
                                  ast::binary_op_type_to_string(node->type));
    }

    switch (node->type) {
        case ast::BinaryOpType::LT:
        case ast::BinaryOpType::LTE:
        case ast::BinaryOpType::GT:
        case ast::BinaryOpType::GTE:
            check_numeric(lhs, line);
            check_numeric(rhs, line);
            node->set_type(ctx::bool_);
            break;
        case ast::BinaryOpType::ADD:
        case ast::BinaryOpType::SUB:
        case ast::BinaryOpType::MUL:
        case ast::BinaryOpType::DIV:
        case ast::BinaryOpType::POW:
        case ast::BinaryOpType::MOD:
            check_numeric(lhs, line);
            check_numeric(rhs, line);
            node->set_type(node->lhs->get_type());
            break;
        case ast::BinaryOpType::AND:
        case ast::BinaryOpType::OR:
            check_bool(lhs, line);
            check_bool(rhs, line);
            node->set_type(ctx::bool_);
            break;
        case ast::BinaryOpType::EQ:
        case ast::BinaryOpType::NE:
            node->set_type(ctx::bool_);
            break;
    }
}

void TypeCheck::visit_unary_operator(shared_ptr<ast::UnaryOperator> node) {
    visit(node->rhs);

    size_t line = node->token->getLine();
    TypePtr rhs = node->get_type();
    if (node->type == ast::UnaryOpType::MINUS && !rhs->is_numeric()) {
        throw TypeError(
            line, "unary minus only works on numeric types, found type: " +
                      rhs->get_name());
    }

    if (node->type == ast::UnaryOpType::NOT && *rhs != *ctx::bool_) {
        throw TypeError(line,
                        "unary minus only works on booleans, found type: " +
                            rhs->get_name());
    }
}

void TypeCheck::visit_conditional(shared_ptr<ast::Conditional> node) {
    visit(node->condition);
    if (*node->condition->get_type() != *ctx::bool_) {
        throw TypeError(node->token->getLine(), "condition must be boolean");
    }

    visit(node->body);

    if (node->else_if.has_value()) {
        visit(node->else_if.value());
    }
}

void TypeCheck::check_numeric(TypePtr type, size_t line) {
    if (!type->is_numeric()) {
        throw TypeError(line, "type(" + type->get_name() + ") is not numeric");
    }
}

void TypeCheck::check_bool(TypePtr type, size_t line) {
    if (*type != *ctx::bool_) {
        throw TypeError(line, "type(" + type->get_name() + ") is not boolean");
    }
}
