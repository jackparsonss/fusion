#include "FusionLexer.h"
#include "FusionParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"

#include <iostream>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  fusion::FusionLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  fusion::FusionParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  antlr4::tree::ParseTree *tree = parser.file();

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);

  // HOW TO WRITE OUT.
  // std::ofstream out(argv[2]);
  // out << "This is out...\n";

  return 0;
}
