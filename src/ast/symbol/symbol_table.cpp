#include <memory>

#include "CommonToken.h"
#include "ast/ast.h"
#include "ast/symbol/function_symbol.h"
#include "ast/symbol/symbol_table.h"
#include "shared/context.h"

namespace {
shared_ptr<ast::Function> make_print(TypePtr type) {
    Token* token = new antlr4::CommonToken(1);
    auto body = make_shared<ast::Block>(token);

    std::vector<shared_ptr<ast::Parameter>> params = {
        make_shared<ast::Parameter>(
            make_shared<ast::Variable>(ast::Qualifier::Let, type, "arg", token),
            token),
    };

    std::string name = "print_" + type->get_name();
    auto print =
        make_shared<ast::Function>(name, name, body, type, params, token);
    return print;
}

shared_ptr<ast::Function> make_println(TypePtr type) {
    Token* token = new antlr4::CommonToken(1);
    auto body = make_shared<ast::Block>(token);

    std::vector<shared_ptr<ast::Parameter>> params = {
        make_shared<ast::Parameter>(
            make_shared<ast::Variable>(ast::Qualifier::Let, type, "arg", token),
            token),
    };

    std::string name = "println_" + type->get_name();
    auto print =
        make_shared<ast::Function>(name, name, body, type, params, token);
    return print;
}
}  // namespace

SymbolTable::SymbolTable() {
    ScopePtr global_scope = make_shared<Scope>(nullptr);
    this->scopes.push_back(global_scope);
    current_scope = global_scope;

    init_types();
    init_builtins();
}

void SymbolTable::init_types() {
    for (const auto& ty : ctx::primitives) {
        define(make_shared<BuiltinTypeSymbol>(ty->get_name()));
    }
}

void SymbolTable::init_builtins() {
    for (const auto& ty : ctx::primitives) {
        auto print = make_print(ty);
        auto println = make_println(ty);

        define(make_shared<FunctionSymbol>(print, this->current_scope));
        define(make_shared<FunctionSymbol>(println, this->current_scope));
    }
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
