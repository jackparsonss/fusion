#include "ast/symbol/symbol_table.h"
#include "CommonToken.h"
#include "ast/symbol/function_symbol.h"
#include "shared/context.h"

SymbolTable::SymbolTable() {
    ScopePtr global_scope = make_shared<Scope>(nullptr);
    this->scopes.push_back(global_scope);
    current_scope = global_scope;

    init_types();
}

void SymbolTable::init_types() {
    Token* token = new antlr4::CommonToken(1);

    define(make_shared<BuiltinTypeSymbol>("i32"));

    auto print_body = make_shared<ast::Block>(token);
    auto print =
        make_shared<ast::Function>("print", print_body, ctx::i32, token);
    define(make_shared<FunctionSymbol>(print, this->current_scope));
}

void SymbolTable::push() {
    ScopePtr scope = make_shared<Scope>(this->scopes.back());
    current_scope = scope;
    this->scopes.back()->enclose_scope(scope);
    this->scopes.push_back(scope);
}

void SymbolTable::pop() {
    this->scopes.pop_back();
    current_scope = this->scopes.back();
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
