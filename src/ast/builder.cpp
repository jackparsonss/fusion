#include <stdexcept>

#include "ast/ast.h"
#include "ast/builder.h"

#define cast_node(a, b) \
    (dynamic_pointer_cast<a>(std::any_cast<shared_ptr<ast::Node>>(b)))
#define to_node(a) static_cast<shared_ptr<ast::Node>>(a)

Builder::Builder(shared_ptr<SymbolTable> symbol_table) {
    this->symbol_table = symbol_table;
}

bool Builder::has_ast() {
    return this->ast != nullptr;
}

shared_ptr<ast::Block> Builder::get_ast() {
    return this->ast;
}

std::any Builder::visitFile(FusionParser::FileContext* ctx) {
    this->ast = std::make_shared<ast::Block>(nullptr);

    for (auto const& s : ctx->statement()) {
        shared_ptr<ast::Node> node = cast_node(ast::Node, visit(s));

        this->ast->nodes.push_back(node);
    }

    return nullptr;
}

std::any Builder::visitStatement(FusionParser::StatementContext* ctx) {
    if (ctx->declaration() != nullptr) {
        return visit(ctx->declaration());
    }

    if (ctx->function() != nullptr) {
        return visit(ctx->function());
    }

    if (ctx->block() != nullptr) {
        return visit(ctx->block());
    }

    if (ctx->call() != nullptr) {
        return visit(ctx->call());
    }

    throw std::runtime_error("found an invalid statement");
}

std::any Builder::visitDeclaration(FusionParser::DeclarationContext* ctx) {
    std::string name = ctx->ID()->getText();
    Token* token = ctx->ID()->getSymbol();

    ast::Qualifier qualifier =
        std::any_cast<ast::Qualifier>(visit(ctx->qualifier()));
    TypePtr type = std::any_cast<TypePtr>(visit(ctx->type()));

    shared_ptr<ast::Expression> expr =
        cast_node(ast::Expression, visit(ctx->expr()));

    auto var = make_shared<ast::Variable>(qualifier, type, name, token);
    auto decl = make_shared<ast::Declaration>(var, expr, token);

    return to_node(decl);
}

std::any Builder::visitType(FusionParser::TypeContext* ctx) {
    auto type = symbol_table->resolve(ctx->getText());
    if (!type.has_value()) {
        throw std::runtime_error("invalid type found");
    }

    return dynamic_pointer_cast<Type>(type.value());
}

std::any Builder::visitQualifier(FusionParser::QualifierContext* ctx) {
    if (ctx->CONST() != nullptr) {
        return ast::Qualifier::Const;
    }

    if (ctx->LET() != nullptr) {
        return ast::Qualifier::Let;
    }

    throw std::runtime_error("invalid qualifier found");
}

std::any Builder::visitLiteralInt(FusionParser::LiteralIntContext* ctx) {
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

std::any Builder::visitVariable(FusionParser::VariableContext* ctx) {
    Token* token = ctx->ID()->getSymbol();
    std::string name = ctx->ID()->getText();
    TypePtr type = make_shared<Type>(Type::unset);

    auto var =
        make_shared<ast::Variable>(ast::Qualifier::Let, type, name, token);

    return to_node(var);
}

std::any Builder::visitBlock(FusionParser::BlockContext* ctx) {
    Token* token = ctx->L_CURLY()->getSymbol();
    auto block = std::make_shared<ast::Block>(token);

    for (auto const& s : ctx->statement()) {
        shared_ptr<ast::Node> node = cast_node(ast::Node, visit(s));

        block->nodes.push_back(node);
    }

    return to_node(block);
}

std::any Builder::visitFunction(FusionParser::FunctionContext* ctx) {
    Token* token = ctx->FUNCTION()->getSymbol();
    TypePtr type = std::any_cast<TypePtr>(visit(ctx->type()));
    std::string name = ctx->ID()->getText();
    auto block = cast_node(ast::Block, visit(ctx->block()));

    auto func = make_shared<ast::Function>(name, block, type, token);
    return to_node(func);
}

std::any Builder::visitCall(FusionParser::CallContext* ctx) {
    Token* token = ctx->L_PAREN()->getSymbol();
    std::string name = ctx->ID()->getText();
    std::vector<shared_ptr<ast::Expression>> args;

    auto call = make_shared<ast::Call>(name, args, token);
    return to_node(call);
}
