#pragma once

#include <memory>
#include <string>

#include "ast/ast.h"
#include "ast/symbol/symbol_table.h"

using std::shared_ptr;

class Pass {
   protected:
    void visit(shared_ptr<ast::Node>);

   public:
    std::string name;
    explicit Pass(std::string name);
    virtual void run(shared_ptr<ast::Block> ast);
    virtual void visit_block(shared_ptr<ast::Block>);
    virtual void visit_integer_literal(shared_ptr<ast::IntegerLiteral>);
    virtual void visit_character_literal(shared_ptr<ast::CharacterLiteral>);
    virtual void visit_boolean_literal(shared_ptr<ast::BooleanLiteral>);
    virtual void visit_declaration(shared_ptr<ast::Declaration>);
    virtual void visit_assignment(shared_ptr<ast::Assignment>);
    virtual void visit_variable(shared_ptr<ast::Variable>);
    virtual void visit_function(shared_ptr<ast::Function>);
    virtual void visit_call(shared_ptr<ast::Call>);
    virtual void visit_parameter(shared_ptr<ast::Parameter>);
    virtual void visit_return(shared_ptr<ast::Return>);
    virtual void visit_binary_operator(shared_ptr<ast::BinaryOperator>);
    virtual void visit_unary_operator(shared_ptr<ast::UnaryOperator>);

    static void run_passes(shared_ptr<ast::Block> ast, shared_ptr<SymbolTable>);
};
