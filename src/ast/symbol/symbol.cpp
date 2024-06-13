#include "ast/symbol/symbol.h"

Symbol::Symbol(std::string name) : Symbol(name, nullptr) {}

Symbol::Symbol(std::string name, std::shared_ptr<Type> type) {
    this->name = name;
    this->type = type;
}

std::string Symbol::get_name() {
    return this->name;
}

std::shared_ptr<Type> Symbol::get_type() {
    return this->type;
}

BuiltinTypeSymbol::BuiltinTypeSymbol(std::string name, NativeType base)
    : Symbol(name), Type(base) {}

std::string BuiltinTypeSymbol::get_name() {
    return Symbol::get_name();
}

VariableSymbol::VariableSymbol(std::string name, std::shared_ptr<Type> type)
    : Symbol(name, type) {}
