#pragma once

#include "FusionBaseVisitor.h"
#include "FusionParser.h"
#include "ast/ast.h"

using namespace fusion;

class AstBuilder : public FusionBaseVisitor {
   private:
    ast::Block* ast;

   public:
    bool has_ast();
    ast::Block* get_ast();

    std::any visitFile(FusionParser::FileContext* ctx) override;
};
