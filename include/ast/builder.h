#pragma once

#include <memory>

#include "FusionBaseVisitor.h"
#include "FusionParser.h"
#include "ast/ast.h"

using namespace fusion;

class AstBuilder : public FusionBaseVisitor {
   private:
    std::shared_ptr<ast::Block> ast;

   public:
    bool has_ast();
    std::shared_ptr<ast::Block> get_ast();

    std::any visitFile(FusionParser::FileContext* ctx) override;
};
