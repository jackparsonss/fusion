#include "ast/passes/def_ref.h"

DefRef::DefRef(shared_ptr<SymbolTable> symbol_table) : Pass("DefRef") {
    this->symbol_table = symbol_table;
}

void DefRef::visit_variable(shared_ptr<ast::Variable> node) {
    std::optional<SymbolPtr> var = symbol_table->resolve(node->get_name());
    if (!var.has_value()) {
        throw std::runtime_error("found undefined variable");
    }

    shared_ptr<VariableSymbol> vs =
        dynamic_pointer_cast<VariableSymbol>(var.value());
    if (vs == nullptr) {
        throw std::runtime_error("found non variable in symbol table");
    }

    node->set_type(vs->variable->get_type());
    node->set_qualifier(vs->variable->get_qualifier());
    node->set_ref_name(vs->variable->get_ref_name());
}

void DefRef::visit_declaration(shared_ptr<ast::Declaration> node) {
    if (symbol_table->resolve_local(node->var->get_name()).has_value()) {
        throw std::runtime_error("variable already defined");
    }

    auto var = node->var;
    shared_ptr<VariableSymbol> sym = std::make_shared<VariableSymbol>(var);

    symbol_table->define(sym);

    visit(node->expr);
}
