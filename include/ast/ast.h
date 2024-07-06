#pragma once

#include <memory>
#include <string>

#include "Token.h"

#include "shared/type/type.h"

using antlr4::Token, std::make_shared, std::shared_ptr;

namespace ast {
enum class Qualifier {
    Const,
    Let,
};

enum class BinaryOpType {
    POW,
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    GT,
    GTE,
    LT,
    LTE,
    EQ,
    NE,
    AND,
    OR,
};

enum class UnaryOpType {
    MINUS,
    NOT,
};

std::string random_name();
std::string qualifier_to_string(Qualifier qualifier);
std::string binary_op_type_to_string(BinaryOpType type);
std::string unary_op_type_to_string(UnaryOpType type);

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
    virtual bool is_l_value();
};

class IntegerLiteral : public Expression {
   private:
    long long value;

   public:
    explicit IntegerLiteral(long long value, TypePtr type, Token* token);
    long long get_value() const;

    void xml(int level) override;
};

class CharacterLiteral : public Expression {
   private:
    char value;

   public:
    explicit CharacterLiteral(char value, Token* token);
    char get_value() const;

    void xml(int level) override;
};

class BooleanLiteral : public Expression {
   private:
    bool value;

   public:
    explicit BooleanLiteral(bool value, Token* token);
    bool get_value() const;

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
    bool is_l_value() override;
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

class Assignment : public Node {
   public:
    shared_ptr<Variable> var;
    shared_ptr<Expression> expr;
    explicit Assignment(shared_ptr<Variable> var,
                        shared_ptr<Expression> expr,
                        Token* token);

    void xml(int level) override;
};

class Parameter : public Node {
   public:
    shared_ptr<Variable> var;
    explicit Parameter(shared_ptr<Variable> var, Token* token);
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
             std::vector<shared_ptr<Parameter>> params,
             Token* token);

    Function(std::string name,
             std::string ref_name,
             shared_ptr<Block> body,
             TypePtr return_type,
             std::vector<shared_ptr<Parameter>> params,
             Token* token);
    shared_ptr<Block> body;
    std::vector<shared_ptr<Parameter>> params;

    std::string get_name();
    std::string get_ref_name();
    void set_ref_name(std::string name);
    void xml(int level) override;
};

class Call : public Expression {
   private:
    std::string name;
    shared_ptr<Function> function;

   public:
    std::vector<shared_ptr<Expression>> arguments;
    Call(std::string name,
         shared_ptr<Function> func,
         std::vector<shared_ptr<Expression>> args,
         Token* token);
    Call(std::string name,
         std::vector<shared_ptr<Expression>> args,
         Token* token);

    std::string get_name();
    void set_function(shared_ptr<Function> func);
    shared_ptr<Function> get_function();
    void xml(int level) override;
};

class Return : public Node {
   public:
    shared_ptr<Expression> expr;
    Return(shared_ptr<Expression> expr, Token* token);
    void xml(int level) override;
};

class BinaryOperator : public Expression {
   public:
    BinaryOpType type;
    shared_ptr<Expression> lhs;
    shared_ptr<Expression> rhs;

    BinaryOperator(BinaryOpType type,
                   shared_ptr<Expression> lhs,
                   shared_ptr<Expression> rhs,
                   Token* token);
    void xml(int level) override;
};

class UnaryOperator : public Expression {
   public:
    UnaryOpType type;
    shared_ptr<Expression> rhs;

    UnaryOperator(UnaryOpType type, shared_ptr<Expression> rhs, Token* token);
    void xml(int level) override;
};

class Conditional : public Node {
   public:
    shared_ptr<Expression> condition;
    shared_ptr<Block> body;
    std::optional<shared_ptr<Conditional>> else_if;

    Conditional(shared_ptr<Expression> condition,
                shared_ptr<Block> body,
                std::optional<shared_ptr<Conditional>> else_if,
                Token* token);
    Conditional(shared_ptr<Expression> condition,
                shared_ptr<Block> body,
                Token* token);

    void xml(int level) override;
};

class Loop : public Node {
   public:
    shared_ptr<Declaration> variable;
    shared_ptr<Expression> condition;
    shared_ptr<Assignment> assignment;
    shared_ptr<Block> body;
    Loop(shared_ptr<Declaration> variable,
         shared_ptr<Expression> condition,
         shared_ptr<Assignment> assignment,
         shared_ptr<Block> body,
         Token* token);

    void xml(int level) override;
};

class Continue : public Node {
   public:
    Continue(Token* token) : Node(token) {}
    void xml(int level);
};

class Break : public Node {
   public:
    Break(Token* token) : Node(token) {}
    void xml(int level);
};
}  // namespace ast
