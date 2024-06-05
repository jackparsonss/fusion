#pragma once

#include "ast/ast.h"
#include "mlir/IR/Value.h"

class BackendVisitor {
   protected:
    mlir::Value visit(ast::Node*);

   public:
    ast::Block* ast;

    explicit BackendVisitor(ast::Block*);
    virtual ast::Block* traverse() = 0;
    virtual mlir::Value visit_block(ast::Block*) = 0;
};
