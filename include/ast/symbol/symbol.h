#pragma once

#include <memory>
#include <string>

#include "ast/ast.h"
#include "shared/type.h"

using std::shared_ptr;

class Symbol {
   private:
    std::string name;
    TypePtr type;

   public:
    virtual std::string get_name();
    virtual TypePtr get_type();

    Symbol(std::string name);
    Symbol(std::string name, TypePtr type);
};

typedef shared_ptr<Symbol> SymbolPtr;

class BuiltinTypeSymbol : public Symbol, public Type {
   public:
    BuiltinTypeSymbol(std::string name);
    std::string get_name() override;
};

class VariableSymbol : public Symbol {
   public:
    shared_ptr<ast::Variable> variable;
    VariableSymbol(shared_ptr<ast::Variable> variable);
};
