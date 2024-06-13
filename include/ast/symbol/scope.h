#pragma once

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>

#include "ast/symbol/symbol.h"

class Scope;

typedef std::shared_ptr<Scope> ScopePtr;

class Scope {
   private:
    ScopePtr enclosing_scope;
    std::deque<ScopePtr> children;
    std::unordered_map<std::string, SymbolPtr> symbol_map;

   public:
    explicit Scope(ScopePtr enclosing_scope);
    ScopePtr get_enclosing_scope();
    void enclose_scope(ScopePtr scope);
    void define(SymbolPtr symbol);
    std::optional<SymbolPtr> resolve(std::string name);
    std::optional<SymbolPtr> resolve_local(std::string name);
};
