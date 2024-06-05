#pragma once

#include "ast/ast.h"
#include "backend/visitor.h"
#include "mlir/IR/Value.h"

class Backend : public BackendVisitor {
    ast::Block* traverse() override;
    mlir::Value visit_block(ast::Block*) override;
};
