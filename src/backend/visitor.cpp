#include "backend/backend.h"

BackendVisitor::BackendVisitor(shared_ptr<ast::Block> ast) {
    this->ast = ast;
}

mlir::Value BackendVisitor::visit(shared_ptr<ast::Node> node) {
    if (const auto block = dynamic_pointer_cast<ast::Block>(node)) {
        return this->visit_block(block);
    }

    if (const auto literal = dynamic_pointer_cast<ast::IntegerLiteral>(node)) {
        return this->visit_integer_literal(literal);
    }

    return nullptr;
}

mlir::Value Backend::visit_block(shared_ptr<ast::Block> node) {
    for (shared_ptr<ast::Node> const& statement : node->nodes) {
        visit(statement);
    }

    return nullptr;
}

mlir::Value Backend::visit_integer_literal(
    shared_ptr<ast::IntegerLiteral> node) {
    return integer::create_i32(node->get_value());
}
