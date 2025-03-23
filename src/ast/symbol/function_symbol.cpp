#include "ast/symbol/function_symbol.h"
#include "ast/ast.h"
#include "ast/symbol/scope.h"
#include "ast/symbol/symbol.h"

FunctionSymbol::FunctionSymbol(shared_ptr<ast::Function> function,
                               ScopePtr enclosing_scope)
    : Symbol(function->get_name(), function->get_type()),
      Scope(enclosing_scope) {
    this->function = function;
}
