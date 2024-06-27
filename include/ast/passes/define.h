#pragma once

#include "ast/ast.h"
#include "ast/passes/pass.h"
#include "ast/symbol/symbol.h"
#include "ast/symbol/symbol_table.h"

class Define : public Pass {
   private:
    shared_ptr<SymbolTable> symbol_table;

   public:
    explicit Define(shared_ptr<SymbolTable> symbol_table);
    void visit_declaration(shared_ptr<ast::Declaration>) override;
    void visit_parameter(shared_ptr<ast::Parameter>) override;
    void visit_function(shared_ptr<ast::Function>) override;
};
