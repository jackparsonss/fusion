#pragma once

#include <memory>
#include <stdexcept>

#include "FusionBaseVisitor.h"
#include "FusionParser.h"

#include "ast/ast.h"
#include "shared/type.h"

using std::make_shared;
using namespace fusion;

class Builder : public FusionBaseVisitor {
   private:
    std::shared_ptr<ast::Block> ast;

   public:
    bool has_ast();
    std::shared_ptr<ast::Block> get_ast();

    std::any visitFile(FusionParser::FileContext* ctx) override;
    std::any visitStatement(FusionParser::StatementContext* ctx) override;
    std::any visitLiteralInt(FusionParser::LiteralIntContext* ctx) override;
    std::any visitDeclaration(FusionParser::DeclarationContext* ctx) override;
    std::any visitType(FusionParser::TypeContext* ctx) override;
    std::any visitQualifier(FusionParser::QualifierContext* ctx) override;
};
