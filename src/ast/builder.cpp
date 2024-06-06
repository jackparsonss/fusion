#include "ast/builder.h"

std::any AstBuilder::visitFile(FusionParser::FileContext* ctx) {
    this->ast = std::make_shared<ast::Block>(nullptr);

    return nullptr;
}

bool AstBuilder::has_ast() {
    return this->ast != nullptr;
}

std::shared_ptr<ast::Block> AstBuilder::get_ast() {
    return this->ast;
}
