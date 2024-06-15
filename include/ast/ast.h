#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "CommonToken.h"
#include "Token.h"

#include "shared/type.h"

using antlr4::Token;
using std::make_shared;
using std::shared_ptr;

namespace ast {
enum class Qualifier {
    Const,
    Let,
};

std::string random_name();
std::string qualifier_to_string(Qualifier qualifier);

class Node {
   public:
    Token* token;

    explicit Node(Token* token);
    virtual void xml(int level) = 0;
};

class Block : public Node {
   public:
    std::vector<std::shared_ptr<Node>> nodes;

    explicit Block(Token* token);
    void xml(int level) override;
};

class Expression : public Node {
   protected:
    TypePtr type;

   public:
    explicit Expression(TypePtr type, Token* token);

    void set_type(TypePtr type);
    TypePtr get_type() const;
};

class IntegerLiteral : public Expression {
   private:
    int value;

   public:
    explicit IntegerLiteral(int value, Token* token);
    int get_value() const;

    void xml(int level) override;
};

class Variable : public Expression {
   private:
    std::string name;
    std::string ref_name;
    Qualifier qualifier;

   public:
    explicit Variable(Qualifier qualifier,
                      TypePtr type,
                      std::string name,
                      Token* token);

    std::string get_ref_name();
    std::string get_name();
    Qualifier get_qualifier();

    void set_qualifier(Qualifier qualifer);
    void set_ref_name(std::string name);
    void xml(int level) override;
};

class Declaration : public Node {
   public:
    shared_ptr<Variable> var;
    shared_ptr<Expression> expr;
    explicit Declaration(shared_ptr<Variable> var,
                         shared_ptr<Expression> expr,
                         Token* token);

    void xml(int level) override;
};

class Function : public Expression {
   private:
    std::string name;
    std::string ref_name;

   public:
    Function(std::string name,
             shared_ptr<Block> body,
             TypePtr return_type,
             Token* token);
    shared_ptr<Block> body;

    std::string get_name();
    std::string get_ref_name();
    void xml(int level) override;
};
}  // namespace ast
