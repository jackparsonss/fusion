#include "FusionLexer.h"
#include "FusionParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "ast/builder.h"
#include "tree/ParseTree.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Missing required argument.\n"
                  << "Required arguments: <input.fuse> <output>\n";
        return 1;
    }

    // Open the file then parse and lex it.
    antlr4::ANTLRFileStream afs;
    afs.loadFromFile(argv[1]);
    fusion::FusionLexer lexer(&afs);
    antlr4::CommonTokenStream tokens(&lexer);
    fusion::FusionParser parser(&tokens);

    antlr4::tree::ParseTree* tree = parser.file();
    AstBuilder builder;
    builder.visit(tree);

    return 0;
}
