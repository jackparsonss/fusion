#include "ast/ast.h"

ast::Node::Node(Token* token) {
    this->token = new antlr4::CommonToken(token);
}

ast::Block::Block(Token* token) : Node(token) {}

void ast::Block::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<block>\n";

    for (Node* const& node : this->nodes) {
        node->xml(level + 1);
        std::cout << std::endl;
    }

    std::cout << std::string(level * 4, ' ') << "</block>\n";
}
