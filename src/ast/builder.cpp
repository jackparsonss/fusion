#include "ast/builder.h"

#define to_node(a, b) \
    (dynamic_pointer_cast<a>(std::any_cast<std::shared_ptr<ast::Node>>(b)))

bool AstBuilder::has_ast() {
    return this->ast != nullptr;
}

std::shared_ptr<ast::Block> AstBuilder::get_ast() {
    return this->ast;
}

std::any AstBuilder::visitFile(FusionParser::FileContext* ctx) {
    this->ast = std::make_shared<ast::Block>(nullptr);

    for (auto const& s : ctx->statement()) {
        // auto n = std::any_cast<std::shared_ptr<ast::Node>>(s);
        // std::shared_ptr<ast::Node> node = to_node(ast::Node, this->visit(s));

        // this->ast->nodes.push_back(node);
    }

    return nullptr;
}

std::any AstBuilder::visitStatement(FusionParser::StatementContext* ctx) {
    if (ctx->expr() != nullptr) {
        return visit(ctx->expr());
    }

    throw std::runtime_error("found an invalid statement");
}

std::any AstBuilder::visitLiteralInt(FusionParser::LiteralIntContext* ctx) {
    Token* token = ctx->INT()->getSymbol();

    // std::shared_ptr<ast::Node> node = std::make_shared<ast::Node>(token);
    // return node;
    return nullptr;
}
