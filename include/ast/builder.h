#pragma once

#include <memory>
#include <stdexcept>

#include "FusionBaseVisitor.h"
#include "FusionParser.h"

#include "ast/ast.h"
#include "ast/symbol/symbol_table.h"
#include "shared/type.h"

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
    std::any visitDeclaration(FusionParser::DeclarationContext* ctx) override;
    std::any visitType(FusionParser::TypeContext* ctx) override;
    std::any visitQualifier(FusionParser::QualifierContext* ctx) override;
    std::any visitVariable(FusionParser::VariableContext* ctx) override;
    std::any visitBlock(FusionParser::BlockContext* ctx) override;
    std::any visitFunction(FusionParser::FunctionContext* ctx) override;
    std::any visitCall(FusionParser::CallContext* ctx) override;
};
