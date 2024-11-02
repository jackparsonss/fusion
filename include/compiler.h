#pragma once

#include <memory>

#include "ast/builder.h"
#include "ast/symbol/symbol_table.h"
#include "backend/backend.h"
#include "errors/syntax.h"
#include "module/manager.h"

using std::shared_ptr, std::unique_ptr;
namespace fs = std::filesystem;

class Compiler {
   private:
    shared_ptr<SymbolTable> symbol_table;
    unique_ptr<Backend> backend;
    unique_ptr<Builder> builder;

    shared_ptr<module::Unit> entry;
    LexerErrorListener* lexer_error;
    SyntaxErrorListener* syntax_error;

   public:
    Compiler(fs::path entry,
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
