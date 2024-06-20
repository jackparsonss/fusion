#include "ast/passes/def_ref.h"
#include "ast/ast.h"
#include "ast/symbol/function_symbol.h"

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
        throw std::runtime_error("variable already defined: " +
                                 node->var->get_name());
    }

    auto var = node->var;
    shared_ptr<VariableSymbol> sym = std::make_shared<VariableSymbol>(var);

    symbol_table->define(sym);

    visit(node->expr);
}

void DefRef::visit_parameter(shared_ptr<ast::Parameter> node) {
    if (symbol_table->resolve_local(node->var->get_name()).has_value()) {
        throw std::runtime_error("parameter already defined: " +
                                 node->var->get_name());
    }

    auto var = node->var;
    shared_ptr<VariableSymbol> sym = std::make_shared<VariableSymbol>(var);
    symbol_table->define(sym);
}

void DefRef::visit_function(shared_ptr<ast::Function> node) {
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

void DefRef::visit_call(shared_ptr<ast::Call> node) {
    for (const auto& arg : node->arguments) {
        visit(arg);
    }

    std::string name = node->get_name();
    if (is_builtin(name) && node->arguments.size() > 0) {
        name += "_" + node->arguments[0]->get_type()->get_name();
    }

    std::optional<SymbolPtr> var = symbol_table->resolve(name);
    if (!var.has_value()) {
        throw std::runtime_error("found undefined function: " + name);
    }

    shared_ptr<FunctionSymbol> vs =
        dynamic_pointer_cast<FunctionSymbol>(var.value());
    if (vs == nullptr) {
        throw std::runtime_error("found non function in symbol table");
    }

    node->set_function(vs->function);
}

bool DefRef::is_builtin(std::string name) {
    if (name == "print") {
        return true;
    }

    return false;
}
