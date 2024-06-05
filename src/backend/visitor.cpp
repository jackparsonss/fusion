#include "backend/visitor.h"

BackendVisitor::BackendVisitor(ast::Block* ast) {
    this->ast = ast;
}

mlir::Value BackendVisitor::visit(ast::Node* node) {
    if (const auto block = dynamic_cast<ast::Block*>(node)) {
        return this->visit_block(block);
    }

    return nullptr;
}
