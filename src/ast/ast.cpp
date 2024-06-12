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

ast::Expression::Expression(TypePtr type, Token* token) : Node(token) {
    this->type = type;
}

void ast::Expression::set_type(TypePtr type) {
    this->type = type;
}

TypePtr ast::Expression::get_type() const {
    return this->type;
}

ast::IntegerLiteral::IntegerLiteral(int value, Token* token)
    : Expression(make_shared<Type>(NativeType::Int), token) {
    this->value = value;
}

int ast::IntegerLiteral::get_value() const {
    return this->value;
}

void ast::IntegerLiteral::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<i32 literal value=\""
              << this->value << "\"/>\n";
}
