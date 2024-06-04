#pragma once

#include "FusionBaseVisitor.h"
#include "FusionParser.h"

using namespace fusion;

class AstBuilder : public FusionBaseVisitor {
    std::any visitFile(FusionParser::FileContext* ctx) override;
};
