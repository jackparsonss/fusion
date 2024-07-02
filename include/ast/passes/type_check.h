#pragma once

#include <stack>
#include "ast/ast.h"
#include "ast/passes/pass.h"
#include "shared/type/type.h"

class TypeCheck : public Pass {
   private:
    std::stack<shared_ptr<ast::Function>> func_stack;
    void check_numeric(TypePtr type, size_t line);
    void check_bool(TypePtr type, size_t line);

   public:
    explicit TypeCheck();
    void visit_declaration(shared_ptr<ast::Declaration>) override;
    void visit_assignment(shared_ptr<ast::Assignment>) override;
    void visit_function(shared_ptr<ast::Function>) override;
    void visit_call(shared_ptr<ast::Call>) override;
    void visit_return(shared_ptr<ast::Return>) override;
    void visit_binary_operator(shared_ptr<ast::BinaryOperator>) override;
    void visit_unary_operator(shared_ptr<ast::UnaryOperator>) override;
};
