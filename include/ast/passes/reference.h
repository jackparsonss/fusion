#pragma once

#include "ast/ast.h"
#include "ast/passes/pass.h"
#include "ast/symbol/symbol.h"
#include "ast/symbol/symbol_table.h"

class Reference : public Pass {
   private:
    shared_ptr<SymbolTable> symbol_table;
    bool is_builtin(std::string name);

   public:
    explicit Reference(shared_ptr<SymbolTable> symbol_table);
    void visit_variable(shared_ptr<ast::Variable>) override;
    void visit_call(shared_ptr<ast::Call>) override;
};
