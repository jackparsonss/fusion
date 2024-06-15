#include "ast/symbol/scope.h"

Scope::Scope(ScopePtr enclosing_scope) {
    this->enclosing_scope = enclosing_scope;
}

ScopePtr Scope::get_enclosing_scope() {
    return this->enclosing_scope;
}

void Scope::enclose_scope(ScopePtr scope) {
    this->children.push_back(scope);
}

void Scope::define(SymbolPtr symbol) {
    this->symbol_map.insert({symbol->get_name(), symbol});
}

std::optional<SymbolPtr> Scope::resolve(std::string name) {
    auto map_return = this->symbol_map.find(name);

    if (map_return != this->symbol_map.end()) {
        return static_cast<SymbolPtr>(map_return->second);
    }

    if (this->get_enclosing_scope() != nullptr) {
        return this->get_enclosing_scope()->resolve(name);
    }

    return std::nullopt;
}

std::optional<SymbolPtr> Scope::resolve_local(std::string name) {
    auto map_return = this->symbol_map.find(name);

    if (map_return != this->symbol_map.end()) {
        return static_cast<SymbolPtr>(map_return->second);
    }

    return std::nullopt;
}
