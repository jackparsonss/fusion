#pragma once

#include <memory>

#include "ANTLRFileStream.h"
#include "FusionLexer.h"
#include "ParseTree.h"
#include "ast/builder.h"
#include "ast/symbol/symbol_table.h"
#include "backend/backend.h"

using std::shared_ptr, std::unique_ptr;

class Compiler {
   private:
    shared_ptr<SymbolTable> symbol_table;
    unique_ptr<Backend> backend;
    unique_ptr<Builder> builder;

    antlr4::ANTLRFileStream* file;
    fusion::FusionLexer* lexer;
    antlr4::tree::ParseTree* tree;
    antlr4::CommonTokenStream* tokens;
    fusion::FusionParser* parser;

   public:
    Compiler(std::string filename,
             shared_ptr<SymbolTable> symbol_table,
             unique_ptr<Backend> backend,
             unique_ptr<Builder> builder);
    ~Compiler();

    void build_ast();
    void run_passes();
    void xml();

    void build_backend();
    void to_object(std::string filename);
    void codegen(std::ofstream& outfile);
};
