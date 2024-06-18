#pragma once

#include "ast/ast.h"
#include "ast/passes/pass.h"
#include "ast/symbol/symbol.h"
#include "ast/symbol/symbol_table.h"

class DefRef : public Pass {
   private:
    shared_ptr<SymbolTable> symbol_table;

   public:
    explicit DefRef(shared_ptr<SymbolTable> symbol_table);
    void visit_variable(shared_ptr<ast::Variable>) override;
    void visit_declaration(shared_ptr<ast::Declaration>) override;
    void visit_function(shared_ptr<ast::Function>) override;
    void visit_call(shared_ptr<ast::Call>) override;
};
