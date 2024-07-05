#include "ast/ast.h"
#include "CommonToken.h"
#include "shared/context.h"

#include <iostream>
#include <random>

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

std::string ast::binary_op_type_to_string(ast::BinaryOpType type) {
    switch (type) {
        case ast::BinaryOpType::POW:
            return "POW";
        case ast::BinaryOpType::ADD:
            return "ADD";
        case ast::BinaryOpType::SUB:
            return "SUB";
        case ast::BinaryOpType::MUL:
            return "MUL";
        case ast::BinaryOpType::DIV:
            return "DIV";
        case ast::BinaryOpType::MOD:
            return "MOD";
        case ast::BinaryOpType::GT:
            return "GT";
        case ast::BinaryOpType::GTE:
            return "GTE";
        case ast::BinaryOpType::LT:
            return "LT";
        case ast::BinaryOpType::LTE:
            return "LTE";
        case ast::BinaryOpType::EQ:
            return "EQ";
        case ast::BinaryOpType::NE:
            return "NE";
        case ast::BinaryOpType::AND:
            return "AND";
        case ast::BinaryOpType::OR:
            return "OR";
    }
}

std::string ast::unary_op_type_to_string(ast::UnaryOpType type) {
    switch (type) {
        case ast::UnaryOpType::MINUS:
            return "MINUS";
        case ast::UnaryOpType::NOT:
            return "NOT";
    }
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

bool ast::Expression::is_l_value() {
    return false;
}

void ast::Expression::set_type(TypePtr type) {
    this->type = type;
}

TypePtr ast::Expression::get_type() const {
    return this->type;
}

ast::BooleanLiteral::BooleanLiteral(bool value, Token* token)
    : Expression(ctx::bool_, token) {
    this->value = value;
}

bool ast::BooleanLiteral::get_value() const {
    return this->value;
}

void ast::BooleanLiteral::xml(int level) {
    std::string s = value ? "true" : "false";
    std::cout << std::string(level * 4, ' ') << "<bool value=\"" << s
              << "\"/>\n";
}

ast::CharacterLiteral::CharacterLiteral(char value, Token* token)
    : Expression(ctx::ch, token) {
    this->value = value;
}

char ast::CharacterLiteral::get_value() const {
    return this->value;
}

void ast::CharacterLiteral::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<ch value=\"" << value
              << "\"/>\n";
}

ast::IntegerLiteral::IntegerLiteral(long long value, TypePtr type, Token* token)
    : Expression(type, token) {
    assert(*type == *ctx::i32 || *type == *ctx::i64);
    this->value = value;
}

long long ast::IntegerLiteral::get_value() const {
    return this->value;
}

void ast::IntegerLiteral::xml(int level) {
    std::cout << std::string(level * 4, ' ')
              << "<" + type->get_name() + " value=\"" << value << "\"/>\n";
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

ast::Qualifier ast::Variable::get_qualifier() {
    return this->qualifier;
}

void ast::Variable::set_qualifier(Qualifier qualifier) {
    this->qualifier = qualifier;
}

void ast::Variable::set_ref_name(std::string name) {
    this->ref_name = name;
}

std::string ast::Variable::get_name() {
    return this->name;
}

std::string ast::Variable::get_ref_name() {
    return this->ref_name;
}

bool ast::Variable::is_l_value() {
    return this->qualifier == ast::Qualifier::Let;
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

ast::Assignment::Assignment(shared_ptr<Variable> var,
                            shared_ptr<Expression> expr,
                            Token* token)
    : Node(token) {
    this->var = var;
    this->expr = expr;
}

void ast::Assignment::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<assignment>\n";
    this->var->xml(level + 1);

    std::cout << std::string((level + 1) * 4, ' ') << "<rhs>\n";
    this->expr->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</rhs>\n";

    std::cout << std::string(level * 4, ' ') << "</assignment>\n";
}

ast::Parameter::Parameter(shared_ptr<Variable> var, Token* token)
    : Node(token) {
    this->var = var;
}

void ast::Parameter::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<parameter>\n";
    this->var->xml(level + 1);
    std::cout << std::string(level * 4, ' ') << "</parameter>\n";
}

ast::Function::Function(std::string name,
                        shared_ptr<Block> body,
                        TypePtr return_type,
                        std::vector<shared_ptr<ast::Parameter>> params,
                        Token* token)
    : Expression(return_type, token) {
    this->name = name;

    if (name == "main") {
        this->ref_name = "main";
    } else {
        this->ref_name = random_name();
    }
    this->body = body;
    this->params = params;
}

ast::Function::Function(std::string name,
                        std::string ref_name,
                        shared_ptr<Block> body,
                        TypePtr return_type,
                        std::vector<shared_ptr<ast::Parameter>> params,
                        Token* token)
    : Function(name, body, return_type, params, token) {
    this->ref_name = ref_name;
}

std::string ast::Function::get_name() {
    return this->name;
}

std::string ast::Function::get_ref_name() {
    return this->ref_name;
}

void ast::Function::set_ref_name(std::string name) {
    this->ref_name = name;
}

void ast::Function::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<function return_type=\""
              << type->get_name() << "\" name=\"" << name << "\" ref_name=\""
              << ref_name << "\">\n";

    if (params.size() > 0) {
        std::cout << std::string((level + 1) * 4, ' ') << "<parameters>\n";
        for (const auto& param : params) {
            param->xml(level + 2);
        }
        std::cout << std::string((level + 1) * 4, ' ') << "</parameters>\n";
    }

    body->xml(level + 1);
    std::cout << std::string(level * 4, ' ') << "</function>\n";
}

ast::Call::Call(std::string name,
                shared_ptr<Function> func,
                std::vector<shared_ptr<Expression>> args,
                Token* token)
    : Expression(func->get_type(), token) {
    this->name = name;
    this->function = func;
    this->arguments = args;
}

ast::Call::Call(std::string name,
                std::vector<shared_ptr<Expression>> args,
                Token* token)
    : Expression(make_shared<Unset>(), token) {
    this->name = name;
    this->arguments = args;
    this->function = nullptr;
}

std::string ast::Call::get_name() {
    return this->name;
}

void ast::Call::xml(int level) {
    std::string ref = function ? function->get_ref_name() : "(unset)";
    std::cout << std::string(level * 4, ' ') << "<call name=\"" << name
              << "\" ref_name=\"" << ref << "\" type=\"" << type->get_name()
              << "\">\n";

    if (arguments.size() > 0) {
        std::cout << std::string((level + 1) * 4, ' ') << "<arguments>\n";
        for (auto const& a : arguments) {
            a->xml(level + 2);
        }
        std::cout << std::string((level + 1) * 4, ' ') << "</arguments>\n";
    }

    std::cout << std::string(level * 4, ' ') << "</call>\n";
}

void ast::Call::set_function(shared_ptr<Function> func) {
    this->function = func;
    this->set_type(func->get_type());
}

shared_ptr<ast::Function> ast::Call::get_function() {
    return this->function;
}

ast::Return::Return(shared_ptr<Expression> expr, Token* token) : Node(token) {
    this->expr = expr;
}

void ast::Return::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<return>\n";
    this->expr->xml(level + 1);
    std::cout << std::string(level * 4, ' ') << "</return>\n";
}

ast::BinaryOperator::BinaryOperator(BinaryOpType type,
                                    shared_ptr<Expression> lhs,
                                    shared_ptr<Expression> rhs,
                                    Token* token)
    : Expression(ctx::any, token) {
    this->type = type;
    this->lhs = lhs;
    this->rhs = rhs;
}

void ast::BinaryOperator::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<binop op_type=\""
              << ast::binary_op_type_to_string(type) << "\" type=\""
              << get_type()->get_name() << "\">\n";

    std::cout << std::string((level + 1) * 4, ' ') << "<lhs>\n";
    this->lhs->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</lhs>\n";

    std::cout << std::string((level + 1) * 4, ' ') << "<rhs>\n";
    this->rhs->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</rhs>\n";

    std::cout << std::string(level * 4, ' ') << "</binary operator>\n";
}

ast::UnaryOperator::UnaryOperator(UnaryOpType type,
                                  shared_ptr<Expression> rhs,
                                  Token* token)
    : Expression(rhs->get_type(), token) {
    this->type = type;
    this->rhs = rhs;
}

void ast::UnaryOperator::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<binop op_type=\""
              << ast::unary_op_type_to_string(type) << "\" type=\""
              << get_type()->get_name() << "\">\n";

    std::cout << std::string((level + 1) * 4, ' ') << "<rhs>\n";
    this->rhs->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</rhs>\n";

    std::cout << std::string(level * 4, ' ') << "</binary operator>\n";
}

ast::Conditional::Conditional(shared_ptr<Expression> condition,
                              shared_ptr<Block> body,
                              std::optional<shared_ptr<Conditional>> else_if,
                              Token* token)
    : Node(token) {
    this->condition = condition;
    this->body = body;
    this->else_if = else_if;
}

ast::Conditional::Conditional(shared_ptr<Expression> condition,
                              shared_ptr<Block> body,
                              Token* token)
    : ast::Conditional(condition, body, std::nullopt, token) {}

void ast::Conditional::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<conditional>\n";
    std::cout << std::string((level + 1) * 4, ' ') << "<if>\n";
    condition->xml(level + 2);
    body->xml(level + 2);
    std::cout << std::string((level + 1) * 4, ' ') << "</if>\n";

    if (else_if.has_value()) {
        std::cout << std::string((level + 1) * 4, ' ') << "<else if>\n";
        else_if.value()->xml(level + 2);
        std::cout << std::string((level + 1) * 4, ' ') << "</else if>\n";
    }

    std::cout << std::string(level * 4, ' ') << "</conditional>\n";
}

ast::Loop::Loop(shared_ptr<Declaration> variable,
                shared_ptr<Expression> condition,
                shared_ptr<Assignment> assignment,
                shared_ptr<Block> body,
                Token* token)
    : Node(token) {
    this->variable = variable;
    this->condition = condition;
    this->assignment = assignment;
    this->body = body;
}

void ast::Loop::xml(int level) {
    std::cout << std::string(level * 4, ' ') << "<loop>\n";
    variable->xml(level + 1);
    condition->xml(level + 1);
    assignment->xml(level + 1);
    body->xml(level + 1);
    std::cout << std::string(level * 4, ' ') << "</loop>\n";
}
