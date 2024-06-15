#include "ast/symbol/symbol.h"

Symbol::Symbol(std::string name) : Symbol(name, nullptr) {}

Symbol::Symbol(std::string name, TypePtr type) {
    this->name = name;
    this->type = type;
}

std::string Symbol::get_name() {
    return this->name;
}

std::shared_ptr<Type> Symbol::get_type() {
    return this->type;
}

BuiltinTypeSymbol::BuiltinTypeSymbol(std::string name)
    : Symbol(name), Type(name) {}

std::string BuiltinTypeSymbol::get_name() {
    return Symbol::get_name();
}

VariableSymbol::VariableSymbol(shared_ptr<ast::Variable> variable)
    : Symbol(variable->get_name(), variable->get_type()) {
    this->variable = variable;
}
