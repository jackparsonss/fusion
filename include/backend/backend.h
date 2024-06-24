#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "ast/ast.h"

#include "mlir/IR/Value.h"

using std::shared_ptr;

class Backend {
   private:
    std::unordered_map<std::string, mlir::Value> variables;
    mlir::Value visit(shared_ptr<ast::Node>);

   public:
    shared_ptr<ast::Block> ast;
    explicit Backend(shared_ptr<ast::Block>);

    void codegen(std::ostream& outstream);
    void to_object(std::string filename);

    shared_ptr<ast::Block> traverse();
    mlir::Value visit_block(shared_ptr<ast::Block>);
    mlir::Value visit_integer_literal(shared_ptr<ast::IntegerLiteral>);
    mlir::Value visit_variable(shared_ptr<ast::Variable>);
    mlir::Value visit_declaration(shared_ptr<ast::Declaration>);
    mlir::Value visit_function(shared_ptr<ast::Function>);
    mlir::Value visit_call(shared_ptr<ast::Call>);
    mlir::Value visit_parameter(shared_ptr<ast::Parameter>);
    mlir::Value visit_return(shared_ptr<ast::Return>);
    mlir::Value visit_character_literal(shared_ptr<ast::CharacterLiteral>);
    mlir::Value visit_binary_operator(shared_ptr<ast::BinaryOperator>);
};
