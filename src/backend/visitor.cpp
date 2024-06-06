#include "backend/visitor.h"
#include "backend/backend.h"

BackendVisitor::BackendVisitor(std::shared_ptr<ast::Block> ast) {
    this->ast = ast;
}

mlir::Value BackendVisitor::visit(std::shared_ptr<ast::Node> node) {
    if (const auto block = dynamic_pointer_cast<ast::Block>(node)) {
        return this->visit_block(block);
    }

    return nullptr;
}

mlir::Value Backend::visit_block(std::shared_ptr<ast::Block> node) {
    for (std::shared_ptr<ast::Node> const& statement : node->nodes) {
        visit(statement);
    }

    return nullptr;
}
