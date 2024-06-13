#pragma once

#include <memory>

#include "ast/ast.h"
#include "mlir/IR/Value.h"

class BackendVisitor {
   protected:
    mlir::Value visit(std::shared_ptr<ast::Node>);

   public:
    std::shared_ptr<ast::Block> ast;

    explicit BackendVisitor(std::shared_ptr<ast::Block>);
    virtual std::shared_ptr<ast::Block> traverse() = 0;
    virtual mlir::Value visit_block(std::shared_ptr<ast::Block>) = 0;
    virtual mlir::Value visit_integer_literal(
        std::shared_ptr<ast::IntegerLiteral>) = 0;
};
