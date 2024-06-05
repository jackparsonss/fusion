#pragma once

#include <iostream>
#include <string>

#include "CommonToken.h"
#include "Token.h"

using antlr4::Token;

namespace ast {
class Node {
   public:
    Token* token;

    explicit Node(Token* token);
    virtual void xml(int level) = 0;
};

class Block : public Node {
   public:
    std::vector<Node*> nodes;

    explicit Block(Token* token);
    void xml(int level) override;
};
}  // namespace ast
