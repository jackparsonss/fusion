#include "FusionLexer.h"
#include "FusionParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "ast/builder.h"
#include "backend/backend.h"
#include "tree/ParseTree.h"

#include <iostream>

int main(int argc, char** argv) {
    antlr4::ANTLRFileStream afs;
    afs.loadFromFile(argv[1]);
    fusion::FusionLexer lexer(&afs);
    antlr4::CommonTokenStream tokens(&lexer);
    fusion::FusionParser parser(&tokens);

    antlr4::tree::ParseTree* tree = parser.file();
    AstBuilder builder;
    builder.visit(tree);
    assert(builder.has_ast());

    Backend backend = Backend(builder.get_ast());
    backend.traverse();

    for (unsigned i = 0; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--post-pass") {
            builder.get_ast()->xml(0);
        }

        if (arg == "-o" && i == argc - 1 && argv[i + 1][0] != '-') {
            std::cerr << "You must provide a file to target" << std::endl;
            return 1;
        }

        if (arg == "--emit-llvm" && i == argc - 1 && argv[i + 1][0] != '-') {
            std::cerr << "You must provide a file to target llvm ir"
                      << std::endl;
            return 1;
        }

        if (arg == "-o") {
            std::string filename = std::string(argv[i + 1]);
            backend.to_object(filename + ".o");

            std::string command = "clang " + filename + ".o -o " + filename +
                                  " -L./ -lgazrt -Wl,-rpath,./";
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
