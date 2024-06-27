#include "ast/passes/type_check.h"
#include "ast/ast.h"
#include "errors.h"
#include "shared/context.h"

TypeCheck::TypeCheck() : Pass("Typecheck") {}

void TypeCheck::visit_declaration(shared_ptr<ast::Declaration> node) {
    visit(node->var);
    visit(node->expr);

    Type var = *node->var->get_type();
    Type expr = *node->expr->get_type();
    if (var != expr) {
        throw TypeError(node->token->getLine(),
                        "mismatched lhs(" + var.get_name() + ") and rhs(" +
                            expr.get_name() + ") types on declaration");
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
    for (size_t i = 0; i < node->arguments.size(); i++) {
        visit(node->arguments[i]);

        Type arg = *node->arguments[i]->get_type();
        Type param = *node->get_function()->params[i]->var->get_type();
        if (arg != param) {
            throw TypeError(node->token->getLine(),
                            "mismatched argument(" + arg.get_name() +
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
    Type lhs = *node->lhs->get_type();
    Type rhs = *node->rhs->get_type();
    if (lhs != rhs) {
        throw TypeError(line, "mismatched lhs(" + lhs.get_name() +
                                  ") and rhs(" + rhs.get_name() +
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
            node->set_type(ctx::t_bool);
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
            node->set_type(ctx::t_bool);
            break;
        case ast::BinaryOpType::EQ:
        case ast::BinaryOpType::NE:
            node->set_type(ctx::t_bool);
    }
}

void TypeCheck::visit_unary_operator(shared_ptr<ast::UnaryOperator> node) {
    visit(node->rhs);

    size_t line = node->token->getLine();
    Type rhs = *node->get_type();
    if (node->type == ast::UnaryOpType::MINUS && rhs != *ctx::i32) {
        throw TypeError(
            line, "unary minus only works on numeric types, found type: " +
                      rhs.get_name());
    }

    if (node->type == ast::UnaryOpType::NOT && rhs != *ctx::t_bool) {
        throw TypeError(line,
                        "unary minus only works on booleans, found type: " +
                            rhs.get_name());
    }
}

void TypeCheck::check_numeric(Type type, size_t line) {
    if (type != *ctx::i32) {
        throw TypeError(line, "type(" + type.get_name() + ") is not numeric");
    }
}

void TypeCheck::check_bool(Type type, size_t line) {
    if (type != *ctx::t_bool) {
        throw TypeError(line, "type(" + type.get_name() + ") is not boolean");
    }
}
