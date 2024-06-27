#include "FusionLexer.h"
#include "FusionParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"

#include "ast/builder.h"
#include "ast/passes/pass.h"
#include "ast/symbol/symbol_table.h"
#include "backend/backend.h"
#include "errors.h"
#include "shared/context.h"

#include <iostream>

int main(int argc, char** argv) {
    antlr4::ANTLRFileStream afs;
    afs.loadFromFile(argv[1]);
    fusion::FusionLexer lexer(&afs);
    antlr4::CommonTokenStream tokens(&lexer);
    fusion::FusionParser parser(&tokens);

    antlr4::tree::ParseTree* tree = parser.file();

    ctx::initialize_context();
    std::shared_ptr<SymbolTable> symtab = std::make_shared<SymbolTable>();

    Builder builder = Builder(symtab);

    try {
        builder.visit(tree);
        assert(builder.has_ast());
        for (size_t i = 0; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if (arg == "--xml") {
                builder.get_ast()->xml(0);
            }
        }

        Pass::run_passes(builder.get_ast(), symtab);
        assert(builder.has_ast());
        for (size_t i = 0; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if (arg == "--pass-xml") {
                builder.get_ast()->xml(0);
            }
        }
    } catch (CompileTimeException const& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    Backend backend = Backend(builder.get_ast());
    backend.traverse();

    // backend args
    for (size_t i = 0; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--emit-llvm" && i == argc - 1) {
            std::cerr << "You must provide a file to target llvm ir"
                      << std::endl;
            return 1;
        }

        if (arg == "-o" && i == argc - 1 && argv[i + 1][0] != '-') {
            std::cerr << "You must provide a file to target" << std::endl;
            return 1;
        }

        if (arg == "-o") {
            std::string filename = std::string(argv[i + 1]);
            backend.to_object(filename + ".o");

            std::string command = "clang " + filename + ".o -o " + filename;
            system(command.c_str());

            command = "rm " + filename + ".o";
            system(command.c_str());
        }

        if (arg == "--emit-llvm") {
            std::ofstream outfile(argv[i + 1]);
            backend.codegen(outfile);
        }
    }

    return 0;
}
