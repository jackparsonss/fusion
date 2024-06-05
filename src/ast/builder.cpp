#include "ast/builder.h"

std::any AstBuilder::visitFile(FusionParser::FileContext* ctx) {
    return nullptr;
}

bool AstBuilder::has_ast() {
    return this->ast != nullptr;
}

ast::Block* AstBuilder::get_ast() {
    return this->ast;
}
