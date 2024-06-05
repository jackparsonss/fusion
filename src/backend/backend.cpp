#include "backend/backend.h"

ast::Block* Backend::traverse() {
    visit(ast);

    // TODO: verify module
    return ast;
}

mlir::Value Backend::visit_block(ast::Block* node) {
    for (ast::Node* const& statement : node->nodes) {
        visit(statement);
    }

    return nullptr;
}
