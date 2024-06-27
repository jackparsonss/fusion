#include "ast/passes/builtin.h"
#include "ast/symbol/function_symbol.h"

Builtin::Builtin(shared_ptr<SymbolTable> symbol_table) : Pass("Builtin") {
    this->symbol_table = symbol_table;
}

void Builtin::visit_call(shared_ptr<ast::Call> node) {
    std::string name = node->get_name();
    if (!is_builtin(name)) {
        return;
    }

    name += "_" + node->arguments[0]->get_type()->get_name();
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

bool Builtin::is_builtin(std::string name) {
    if (name == "print" || name == "println") {
        return true;
    }

    return false;
}
