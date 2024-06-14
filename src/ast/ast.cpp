#include "ast/ast.h"

std::string ast::random_name() {
    std::string s = "_";
    std::random_device rd;
    std::uniform_int_distribution<int> distribution(0, 1000000);

    for (int i = 1; i < 60 + 1; i++) {
        int t = distribution(rd);
        t = t < 0 ? -t : t;
        t = t % 26;

        s += static_cast<char>(t + 97);
    }

    return s;
}

std::string ast::qualifier_to_string(ast::Qualifier qualifier) {
    switch (qualifier) {
        case ast::Qualifier::Const:
            return "const";
        case ast::Qualifier::Let:
            return "let";
    }

    throw std::runtime_error("invalid qualifier case");
}

ast::Node::Node(Token* token) {
    if (token == nullptr) {
        return;
    }

    this->token = new antlr4::CommonToken(token);
}

ast::Block::Block(Token* token) : Node(token) {}

void ast::Block::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<block>\n";

    for (shared_ptr<Node> const& node : this->nodes) {
        node->xml(level + 1);
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
    : Expression(make_shared<Type>(Type::i32), token) {
    this->value = value;
}

int ast::IntegerLiteral::get_value() const {
    return this->value;
}

void ast::IntegerLiteral::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<i32 value=\"" << this->value
              << "\"/>\n";
}

ast::Variable::Variable(Qualifier qualifier,
                        TypePtr type,
                        std::string name,
                        Token* token)
    : Expression(type, token) {
    this->qualifier = qualifier;
    this->name = name;
    this->ref_name = random_name();
}

std::string ast::Variable::get_name() {
    return this->name;
}

std::string ast::Variable::get_ref_name() {
    return this->ref_name;
}

void ast::Variable::xml(int level) {
    std::cout << std::string(level * 4, ' ');
    std::cout << "<variable qualifier=\"" << ast::qualifier_to_string(qualifier)
              << "\" type=\"" << type->get_name() << "\" name=\"" << name
              << "\" ref_name=\"" << ref_name << "\"/>\n";
}

ast::Declaration::Declaration(shared_ptr<Variable> var,
                              shared_ptr<Expression> expr,
                              Token* token)
    : Node(token) {
    this->var = var;
    this->expr = expr;
}

void ast::Declaration::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<declaration>\n";
    this->var->xml(level + 1);

    std::cout << std::string((level + 1) * 4, ' ') << "<rhs>\n";
    this->expr->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</rhs>\n";

    std::cout << std::string(level * 4, ' ') << "</declaration>\n";
}
