#include "ast/passes/define.h"
#include "ast/ast.h"
#include "ast/symbol/function_symbol.h"

Define::Define(shared_ptr<SymbolTable> symbol_table) : Pass("Define") {
    this->symbol_table = symbol_table;
}

void Define::visit_declaration(shared_ptr<ast::Declaration> node) {
    if (symbol_table->resolve_local(node->var->get_name()).has_value()) {
        throw std::runtime_error("variable already defined: " +
                                 node->var->get_name());
    }

    auto var = node->var;
    shared_ptr<VariableSymbol> sym = std::make_shared<VariableSymbol>(var);

    symbol_table->define(sym);

    visit(node->expr);
}

void Define::visit_parameter(shared_ptr<ast::Parameter> node) {
    if (symbol_table->resolve_local(node->var->get_name()).has_value()) {
        throw std::runtime_error("parameter already defined: " +
                                 node->var->get_name());
    }

    auto var = node->var;
    shared_ptr<VariableSymbol> sym = std::make_shared<VariableSymbol>(var);
    symbol_table->define(sym);
}

void Define::visit_function(shared_ptr<ast::Function> node) {
    if (symbol_table->resolve_bottom(name).has_value()) {
        throw std::runtime_error("function already defined");
    }

    shared_ptr<FunctionSymbol> sym =
        std::make_shared<FunctionSymbol>(node, symbol_table->current_scope);
    symbol_table->define(sym);

    for (const auto& param : node->params) {
        visit(param);
    }

    visit(node->body);
}
