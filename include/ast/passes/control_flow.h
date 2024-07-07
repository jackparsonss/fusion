#pragma once

#include "ast/passes/pass.h"

class ControlFlow : public Pass {
   private:
    bool in_function;
    bool in_loop;

   public:
    ControlFlow();
    void visit_function(shared_ptr<ast::Function>) override;
    void visit_return(shared_ptr<ast::Return>) override;
    void visit_loop(shared_ptr<ast::Loop>) override;
    void visit_continue(shared_ptr<ast::Continue>) override;
    void visit_break(shared_ptr<ast::Break>) override;
};
