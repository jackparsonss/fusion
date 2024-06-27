#include "ast/passes/reference.h"
#include "ast/symbol/function_symbol.h"

Reference::Reference(shared_ptr<SymbolTable> symbol_table) : Pass("Reference") {
    this->symbol_table = symbol_table;
}

void Reference::visit_variable(shared_ptr<ast::Variable> node) {
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

void Reference::visit_call(shared_ptr<ast::Call> node) {
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

bool Reference::is_builtin(std::string name) {
    if (name == "print" || name == "println") {
        return true;
    }

    return false;
}
