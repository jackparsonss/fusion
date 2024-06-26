#pragma once

#include <deque>

#include "ast/symbol/scope.h"
#include "ast/symbol/symbol.h"

using std::make_shared;

class SymbolTable {
   private:
    void init_types();
    void init_builtins();
    std::deque<ScopePtr> scopes;

   public:
    ScopePtr current_scope;
    SymbolTable();
    void push();
    void pop();
    void define(SymbolPtr symbol);
    void define_bottom(SymbolPtr symbol);

    std::optional<SymbolPtr> resolve(std::string name);
    std::optional<SymbolPtr> resolve_local(std::string name);
    std::optional<SymbolPtr> resolve_bottom(std::string name);
};
