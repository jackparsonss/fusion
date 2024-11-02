#pragma once

#include <string>
#include <unordered_map>
#include "ANTLRFileStream.h"
#include "FusionLexer.h"
#include "FusionParser.h"
#include "ParseTree.h"
#include "ast/ast.h"

using std::shared_ptr;
namespace fs = std::filesystem;

namespace module {
class Unit {
   public:
    antlr4::tree::ParseTree* tree;
    Unit(std::string filename);
    ~Unit();

   private:
    antlr4::ANTLRFileStream* file;
    fusion::FusionLexer* lexer;
    antlr4::CommonTokenStream* tokens;
    fusion::FusionParser* parser;
};

class Manager {
   private:
    std::unordered_map<std::string, shared_ptr<Unit>> module_cache;

   public:
    shared_ptr<Unit> compile_unit(std::string module);
};
};  // namespace module
