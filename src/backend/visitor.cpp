#include "backend/visitor.h"
#include "backend/backend.h"

BackendVisitor::BackendVisitor(ast::Block* ast) {
    this->ast = ast;
}

mlir::Value BackendVisitor::visit(ast::Node* node) {
    if (const auto block = dynamic_cast<ast::Block*>(node)) {
        return this->visit_block(block);
    }

    return nullptr;
}

mlir::Value Backend::visit_block(ast::Block* node) {
    for (ast::Node* const& statement : node->nodes) {
        visit(statement);
    }

    return nullptr;
}
