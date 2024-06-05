#pragma once

#include <ostream>
#include "ast/ast.h"
#include "backend/visitor.h"
#include "mlir/IR/Value.h"

class Backend : public BackendVisitor {
   public:
    explicit Backend(ast::Block*);

    void codegen(std::ostream& outstream);
    void to_object(std::string filename);

    ast::Block* traverse() override;
    mlir::Value visit_block(ast::Block*) override;
};
