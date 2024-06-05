#include "backend/backend.h"

Backend::Backend(ast::Block* ast) : BackendVisitor(ast) {}

ast::Block* Backend::traverse() {
    visit(ast);

    // TODO: verify module
    return ast;
}

void Backend::to_object(std::string filename) {}

void Backend::codegen(std::ostream& outstream) {}
