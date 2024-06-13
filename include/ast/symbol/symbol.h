#pragma once

#include <memory>
#include <string>

#include "shared/type.h"

class Symbol {
   private:
    std::string name;
    std::shared_ptr<Type> type;

   public:
    virtual std::string get_name();
    virtual std::shared_ptr<Type> get_type();

    Symbol(std::string name);
    Symbol(std::string name, std::shared_ptr<Type> type);
};

typedef std::shared_ptr<Symbol> SymbolPtr;

class BuiltinTypeSymbol : public Symbol, public Type {
   public:
    BuiltinTypeSymbol(std::string name, NativeType base);
    std::string get_name() override;
};

class VariableSymbol : public Symbol {
   public:
    VariableSymbol(std::string name, std::shared_ptr<Type> type);
};
