#pragma once

#include <iostream>
#include <memory>
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
   private:
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
}  // namespace ast
