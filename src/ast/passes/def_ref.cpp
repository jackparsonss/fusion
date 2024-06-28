#include "ast/passes/def_ref.h"
#include "ast/ast.h"
#include "ast/symbol/function_symbol.h"
#include "errors/errors.h"

DefRef::DefRef(shared_ptr<SymbolTable> symbol_table) : Pass("DefRef") {
    this->symbol_table = symbol_table;
}

void DefRef::visit_block(shared_ptr<ast::Block> node) {
    symbol_table->push();
    for (const auto& node : node->nodes) {
        visit(node);
    }
    symbol_table->pop();
}

void DefRef::visit_declaration(shared_ptr<ast::Declaration> node) {
    if (symbol_table->resolve_local(node->var->get_name()).has_value()) {
        throw SymbolError(
            node->token->getLine(),
            "variable " + node->var->get_name() + " already defined");
    }

    shared_ptr<ast::Variable> var = node->var;
    shared_ptr<VariableSymbol> sym = make_shared<VariableSymbol>(var);

    symbol_table->define(sym);

    visit(node->expr);
}

void DefRef::visit_assignment(shared_ptr<ast::Assignment> node) {
    visit(node->var);
    if (!node->var->is_l_value()) {
        throw AssignError(
            node->token->getLine(),
            "Cannot assign to const variable " + node->var->get_name());
    }

    visit(node->expr);
}

void DefRef::visit_parameter(shared_ptr<ast::Parameter> node) {
    shared_ptr<ast::Variable> var = node->var;
    if (symbol_table->resolve_local(var->get_name()).has_value()) {
        throw SymbolError(
            node->token->getLine(),
            "parameter " + node->var->get_name() + " already defined");
    }

    shared_ptr<VariableSymbol> sym = make_shared<VariableSymbol>(var);
    symbol_table->define(sym);
}

void DefRef::visit_function(shared_ptr<ast::Function> node) {
    if (symbol_table->resolve_bottom(name).has_value()) {
        throw SymbolError(node->token->getLine(),
                          "function " + node->get_name() + " already defined");
    }

    shared_ptr<FunctionSymbol> sym =
        std::make_shared<FunctionSymbol>(node, symbol_table->current_scope);
    symbol_table->define(sym);

    symbol_table->push();
    for (const auto& param : node->params) {
        visit(param);
    }

    symbol_table->push();
    visit(node->body);
    symbol_table->pop();
    symbol_table->pop();
}

void DefRef::visit_variable(shared_ptr<ast::Variable> node) {
    std::optional<SymbolPtr> var = symbol_table->resolve(node->get_name());
    size_t line = node->token->getLine();
    if (!var.has_value()) {
        throw SymbolError(line,
                          "use of undefined variable: " + node->get_name());
    }

    shared_ptr<VariableSymbol> vs =
        dynamic_pointer_cast<VariableSymbol>(var.value());
    if (vs == nullptr) {
        throw SymbolError(line, node->get_name() + " is not a variable");
    }

    node->set_type(vs->variable->get_type());
    node->set_qualifier(vs->variable->get_qualifier());
    node->set_ref_name(vs->variable->get_ref_name());
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
    size_t line = node->token->getLine();
    if (!var.has_value()) {
        throw SymbolError(line, "called undefined function: " + name);
    }

    shared_ptr<FunctionSymbol> vs =
        dynamic_pointer_cast<FunctionSymbol>(var.value());
    if (vs == nullptr) {
        throw SymbolError(line, node->get_name() + " is not a function");
    }

    node->set_function(vs->function);
}

bool DefRef::is_builtin(std::string name) {
    if (name == "print" || name == "println") {
        return true;
    }

    return false;
}
