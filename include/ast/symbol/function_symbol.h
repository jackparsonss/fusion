#pragma once

#include "ast/ast.h"
#include "ast/symbol/scope.h"
#include "ast/symbol/symbol.h"

class FunctionSymbol : public Symbol, public Scope {
   public:
    shared_ptr<ast::Function> function;
    FunctionSymbol(shared_ptr<ast::Function> function,
                   ScopePtr enclosing_scope);
};
