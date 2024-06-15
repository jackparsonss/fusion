#pragma once

#include <memory>

#include "ast/ast.h"
#include "ast/symbol/symbol_table.h"

using std::shared_ptr;

class Pass {
   protected:
    void visit(shared_ptr<ast::Node>);

   public:
    virtual void run(shared_ptr<ast::Block> ast);
    virtual void visit_block(shared_ptr<ast::Block>);
    virtual void visit_integer_literal(shared_ptr<ast::IntegerLiteral>);
    virtual void visit_declaration(shared_ptr<ast::Declaration>);
    virtual void visit_variable(shared_ptr<ast::Variable>);
};

namespace pass {
void run_passes(shared_ptr<ast::Block> ast, shared_ptr<SymbolTable>);
}
