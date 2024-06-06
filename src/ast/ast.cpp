#include "ast/ast.h"

ast::Node::Node(Token* token) {
    if (token == nullptr) {
        return;
    }

    this->token = new antlr4::CommonToken(token);
}

ast::Block::Block(Token* token) : Node(token) {}

void ast::Block::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<block>\n";

    for (std::shared_ptr<Node> const& node : this->nodes) {
        node->xml(level + 1);
        std::cout << std::endl;
    }

    std::cout << std::string(level * 4, ' ') << "</block>\n";
}
