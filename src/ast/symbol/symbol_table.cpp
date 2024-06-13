#include "ast/symbol/symbol_table.h"

SymbolTable::SymbolTable() {
    ScopePtr global_scope = make_shared<Scope>(nullptr);
    this->scopes.push_back(global_scope);

    init_types();
}

void SymbolTable::init_types() {
    define(make_shared<BuiltinTypeSymbol>("int"));
}

void SymbolTable::push() {
    ScopePtr scope = make_shared<Scope>(this->scopes.back());
    this->scopes.back()->enclose_scope(scope);
    this->scopes.push_back(scope);
}

void SymbolTable::pop() {
    this->scopes.pop_back();
}

void SymbolTable::define(SymbolPtr symbol) {
    this->scopes.back()->define(symbol);
}

void SymbolTable::define_bottom(SymbolPtr symbol) {
    this->scopes.front()->define(symbol);
}

std::optional<SymbolPtr> SymbolTable::resolve(std::string name) {
    return this->scopes.back()->resolve(name);
}

std::optional<SymbolPtr> SymbolTable::resolve_local(std::string name) {
    return this->scopes.back()->resolve_local(name);
}

std::optional<SymbolPtr> SymbolTable::resolve_bottom(std::string name) {
    return this->scopes.front()->resolve(name);
}
