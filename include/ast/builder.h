#pragma once

#include <memory>

#include "FusionBaseVisitor.h"
#include "FusionParser.h"

#include "ast/ast.h"
#include "ast/symbol/symbol_table.h"

using std::make_shared;
using std::shared_ptr;
using namespace fusion;

class Builder : public FusionBaseVisitor {
   private:
    shared_ptr<ast::Block> ast;
    shared_ptr<SymbolTable> symbol_table;

   public:
    Builder(shared_ptr<SymbolTable> symbol_table);
    bool has_ast();
    shared_ptr<ast::Block> get_ast();

    std::any visitFile(FusionParser::FileContext* ctx) override;
    std::any visitStatement(FusionParser::StatementContext* ctx) override;
    std::any visitLiteralInt(FusionParser::LiteralIntContext* ctx) override;
    std::any visitLiteralChar(FusionParser::LiteralCharContext* ctx) override;
    std::any visitLiteralBool(FusionParser::LiteralBoolContext* ctx) override;
    std::any visitDeclaration(FusionParser::DeclarationContext* ctx) override;
    std::any visitType(FusionParser::TypeContext* ctx) override;
    std::any visitQualifier(FusionParser::QualifierContext* ctx) override;
    std::any visitIdentifier(FusionParser::IdentifierContext* ctx) override;
    std::any visitBlock(FusionParser::BlockContext* ctx) override;
    std::any visitFunction(FusionParser::FunctionContext* ctx) override;
    std::any visitCall(FusionParser::CallContext* ctx) override;
    std::any visitVariable(FusionParser::VariableContext* ctx) override;
    std::any visitReturn(FusionParser::ReturnContext* ctx) override;
    std::any visitCallExpr(FusionParser::CallExprContext* ctx) override;
    std::any visitPower(FusionParser::PowerContext* ctx) override;
    std::any visitMulDivMod(FusionParser::MulDivModContext* ctx) override;
    std::any visitAddSub(FusionParser::AddSubContext* ctx) override;
    std::any visitGtLtCond(FusionParser::GtLtCondContext* ctx) override;
    std::any visitEqNeCond(FusionParser::EqNeCondContext* ctx) override;
    std::any visitAndOrCond(FusionParser::AndOrCondContext* ctx) override;
};
