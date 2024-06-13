#include "ast/builder.h"

#define cast_node(a, b) \
    (dynamic_pointer_cast<a>(std::any_cast<std::shared_ptr<ast::Node>>(b)))
#define to_node(a) static_cast<std::shared_ptr<ast::Node>>(a);

bool AstBuilder::has_ast() {
    return this->ast != nullptr;
}

std::shared_ptr<ast::Block> AstBuilder::get_ast() {
    return this->ast;
}

std::any AstBuilder::visitFile(FusionParser::FileContext* ctx) {
    this->ast = std::make_shared<ast::Block>(nullptr);

    for (auto const& s : ctx->statement()) {
        std::shared_ptr<ast::Node> node = cast_node(ast::Node, this->visit(s));

        this->ast->nodes.push_back(node);
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
    int value = 0;
    try {
        value = std::stoi(ctx->INT()->getText());
    } catch (const std::out_of_range& oor) {
        std::runtime_error(oor.what());
    }

    auto node = make_shared<ast::IntegerLiteral>(value, token);
    return to_node(node);
}
