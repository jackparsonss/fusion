#include "ast/symbol/symbol.h"

Symbol::Symbol(std::string name) : Symbol(name, nullptr) {}

Symbol::Symbol(std::string name, TypePtr type) {
    this->name = name;
    this->type = type;
}

std::string Symbol::get_name() {
    return this->name;
}

TypePtr Symbol::get_type() {
    return this->type;
}

BuiltinTypeSymbol::BuiltinTypeSymbol(TypePtr type)
    : Symbol(type->get_name(), type) {}

std::string BuiltinTypeSymbol::get_name() {
    return Symbol::get_name();
}

VariableSymbol::VariableSymbol(shared_ptr<ast::Variable> variable)
    : Symbol(variable->get_name(), variable->get_type()) {
    this->variable = variable;
}
