#pragma once

#include <memory>

#include "ast/ast.h"
#include "mlir/IR/Value.h"

using std::shared_ptr;

class BackendVisitor {
   protected:
    mlir::Value visit(shared_ptr<ast::Node>);

   public:
    shared_ptr<ast::Block> ast;

    explicit BackendVisitor(shared_ptr<ast::Block>);

    virtual shared_ptr<ast::Block> traverse() = 0;
    virtual mlir::Value visit_block(shared_ptr<ast::Block>) = 0;
    virtual mlir::Value visit_integer_literal(
        shared_ptr<ast::IntegerLiteral>) = 0;
    virtual mlir::Value visit_variable(shared_ptr<ast::Variable>) = 0;
    virtual mlir::Value visit_declaration(shared_ptr<ast::Declaration>) = 0;
    virtual mlir::Value visit_function(shared_ptr<ast::Function>) = 0;
    virtual mlir::Value visit_call(shared_ptr<ast::Call>) = 0;
};
