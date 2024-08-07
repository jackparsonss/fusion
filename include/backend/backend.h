#pragma once

#include <memory>
#include <ostream>
#include <stack>
#include <string>

#include "ast/ast.h"

#include "backend/expressions/global.h"
#include "mlir/IR/Value.h"

using std::shared_ptr;

class Backend {
   private:
    std::unordered_map<std::string, mlir::Value> variables;
    GlobalManager gm;

    // used by continue/break
    std::stack<mlir::Block*> loop_conditions;
    std::stack<mlir::Block*> loop_exits;

    mlir::Value visit(shared_ptr<ast::Node>);
    void declare_globals();

   public:
    explicit Backend();

    void codegen(std::ostream& outstream);
    void to_object(std::string filename);

    shared_ptr<ast::Block> traverse(shared_ptr<ast::Block> ast);
    mlir::Value visit_block(shared_ptr<ast::Block>);
    mlir::Value visit_integer_literal(shared_ptr<ast::IntegerLiteral>);
    mlir::Value visit_character_literal(shared_ptr<ast::CharacterLiteral>);
    mlir::Value visit_boolean_literal(shared_ptr<ast::BooleanLiteral>);
    mlir::Value visit_variable(shared_ptr<ast::Variable>);
    mlir::Value visit_declaration(shared_ptr<ast::Declaration>);
    mlir::Value visit_assignment(shared_ptr<ast::Assignment>);
    mlir::Value visit_function(shared_ptr<ast::Function>);
    mlir::Value visit_call(shared_ptr<ast::Call>);
    mlir::Value visit_parameter(shared_ptr<ast::Parameter>);
    mlir::Value visit_return(shared_ptr<ast::Return>);
    mlir::Value visit_binary_operator(shared_ptr<ast::BinaryOperator>);
    mlir::Value visit_unary_operator(shared_ptr<ast::UnaryOperator>);
    mlir::Value visit_conditional(shared_ptr<ast::Conditional>);
    mlir::Value visit_loop(shared_ptr<ast::Loop>);
    mlir::Value visit_continue(shared_ptr<ast::Continue>);
    mlir::Value visit_break(shared_ptr<ast::Break>);
};
