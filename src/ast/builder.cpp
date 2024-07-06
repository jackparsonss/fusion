#include <any>
#include <stdexcept>
#include <vector>

#include "FusionParser.h"
#include "ast/ast.h"
#include "ast/builder.h"
#include "errors/errors.h"
#include "shared/context.h"
#include "shared/type/type.h"

using std::any_cast;

#define cast_node(a, b) \
    (dynamic_pointer_cast<a>(any_cast<shared_ptr<ast::Node>>(b)))
#define to_node(a) static_cast<shared_ptr<ast::Node>>(a)

Builder::Builder(shared_ptr<SymbolTable> symbol_table) {
    this->symbol_table = symbol_table;
}

bool Builder::has_ast() {
    return this->ast != nullptr;
}

shared_ptr<ast::Block> Builder::get_ast() {
    return this->ast;
}

std::any Builder::visitFile(FusionParser::FileContext* ctx) {
    this->ast = std::make_shared<ast::Block>(nullptr);

    for (auto const& s : ctx->statement()) {
        shared_ptr<ast::Node> node = cast_node(ast::Node, visit(s));

        this->ast->nodes.push_back(node);
    }

    return nullptr;
}

std::any Builder::visitStatement(FusionParser::StatementContext* ctx) {
    if (ctx->declaration() != nullptr) {
        return visit(ctx->declaration());
    }

    if (ctx->assignment() != nullptr) {
        return visit(ctx->assignment());
    }

    if (ctx->function() != nullptr) {
        return visit(ctx->function());
    }

    if (ctx->block() != nullptr) {
        return visit(ctx->block());
    }

    if (ctx->call() != nullptr) {
        return visit(ctx->call());
    }

    if (ctx->return_() != nullptr) {
        return visit(ctx->return_());
    }

    if (ctx->if_() != nullptr) {
        return visit(ctx->if_());
    }

    if (ctx->loop() != nullptr) {
        return visit(ctx->loop());
    }

    if (ctx->CONTINUE() != nullptr) {
        auto node = make_shared<ast::Continue>(ctx->CONTINUE()->getSymbol());
        return to_node(node);
    }

    if (ctx->BREAK() != nullptr) {
        auto node = make_shared<ast::Break>(ctx->BREAK()->getSymbol());
        return to_node(node);
    }

    throw std::runtime_error("found an invalid statement");
}

std::any Builder::visitDeclaration(FusionParser::DeclarationContext* ctx) {
    Token* token = ctx->EQUAL()->getSymbol();

    auto expr = cast_node(ast::Expression, visit(ctx->expr()));
    auto var = cast_node(ast::Variable, visit(ctx->variable()));
    auto decl = make_shared<ast::Declaration>(var, expr, token);

    return to_node(decl);
}

std::any Builder::visitType(FusionParser::TypeContext* ctx) {
    auto type = symbol_table->resolve(ctx->getText());
    if (!type.has_value()) {
        throw std::runtime_error("invalid type found");
    }

    return dynamic_pointer_cast<Type>(type.value()->get_type());
}

std::any Builder::visitQualifier(FusionParser::QualifierContext* ctx) {
    if (ctx->CONST() != nullptr) {
        return ast::Qualifier::Const;
    }

    if (ctx->LET() != nullptr) {
        return ast::Qualifier::Let;
    }

    throw std::runtime_error("invalid qualifier found");
}

std::any Builder::visitLiteralInt(FusionParser::LiteralIntContext* ctx) {
    Token* token = ctx->INT()->getSymbol();
    std::string str = ctx->INT()->getText();

    try {
        int num = std::stoi(str);
        auto node = make_shared<ast::IntegerLiteral>(num, ctx::i32, token);
        return to_node(node);
    } catch (const std::out_of_range&) {
        try {
            long long num = std::stoll(str);
            auto node = make_shared<ast::IntegerLiteral>(num, ctx::i64, token);
            return to_node(node);
        } catch (const std::out_of_range&) {
            throw SyntaxError(token->getLine(), "number overflowed 64 bits");
        }
    }

    throw std::runtime_error("failed to parse integer");
}

std::any Builder::visitLiteralBool(FusionParser::LiteralBoolContext* ctx) {
    Token* token = ctx->BOOLEAN()->getSymbol();
    bool value = true;
    if (ctx->BOOLEAN()->getText() == "false") {
        value = false;
    }

    auto node = make_shared<ast::BooleanLiteral>(value, token);
    return to_node(node);
}

std::any Builder::visitLiteralChar(FusionParser::LiteralCharContext* ctx) {
    std::unordered_map<std::string, char> special_characters = {
        {"\\0", '\0'},  {"\\a", '\a'},  {"\\b", '\b'},
        {"\\t", '\t'},  {"\\n", '\n'},  {"\\r", '\r'},
        {"\\\"", '\"'}, {"\\\'", '\''}, {"\\\\", '\\'},
    };
    Token* token = ctx->CHARACTER()->getSymbol();
    std::string literal = ctx->CHARACTER()->getText();
    char value;

    if (literal.size() == 3) {
        value = literal[1];
    } else {
        value = special_characters[literal.substr(1, 2)];
    }

    auto node = make_shared<ast::CharacterLiteral>(value, token);
    return to_node(node);
}

std::any Builder::visitIdentifier(FusionParser::IdentifierContext* ctx) {
    Token* token = ctx->ID()->getSymbol();
    std::string name = ctx->ID()->getText();
    TypePtr type = make_shared<Unset>();

    auto var =
        make_shared<ast::Variable>(ast::Qualifier::Let, type, name, token);

    return to_node(var);
}

std::any Builder::visitBlock(FusionParser::BlockContext* ctx) {
    Token* token = ctx->L_CURLY()->getSymbol();
    auto block = make_shared<ast::Block>(token);

    for (auto const& s : ctx->statement()) {
        shared_ptr<ast::Node> node = cast_node(ast::Node, visit(s));

        block->nodes.push_back(node);
    }

    return to_node(block);
}

std::any Builder::visitVariable(FusionParser::VariableContext* ctx) {
    Token* token = ctx->ID()->getSymbol();
    std::string name = ctx->ID()->getText();

    ast::Qualifier qualifier =
        any_cast<ast::Qualifier>(visit(ctx->qualifier()));
    TypePtr type = any_cast<TypePtr>(visit(ctx->type()));

    auto var = make_shared<ast::Variable>(qualifier, type, name, token);
    return to_node(var);
}

std::any Builder::visitFunction(FusionParser::FunctionContext* ctx) {
    Token* token = ctx->FUNCTION()->getSymbol();
    TypePtr type = any_cast<TypePtr>(visit(ctx->type()));
    std::string name = ctx->ID()->getText();
    auto block = cast_node(ast::Block, visit(ctx->block()));

    auto vars = ctx->variable();
    std::vector<shared_ptr<ast::Parameter>> params(vars.size());
    for (size_t i = 0; i < vars.size(); i++) {
        auto var = cast_node(ast::Variable, visit(vars[i]));
        params[i] = make_shared<ast::Parameter>(var, token);
    }

    auto func = make_shared<ast::Function>(name, block, type, params, token);
    return to_node(func);
}

std::any Builder::visitCall(FusionParser::CallContext* ctx) {
    Token* token = ctx->L_PAREN()->getSymbol();
    std::string name = ctx->ID()->getText();
    std::vector<shared_ptr<ast::Expression>> args;

    for (const auto& arg : ctx->expr()) {
        args.push_back(cast_node(ast::Expression, visit(arg)));
    }

    auto call = make_shared<ast::Call>(name, args, token);
    return to_node(call);
}

std::any Builder::visitCallExpr(FusionParser::CallExprContext* ctx) {
    return visit(ctx->call());
}

std::any Builder::visitReturn(FusionParser::ReturnContext* ctx) {
    Token* token = ctx->RETURN()->getSymbol();
    auto expr = cast_node(ast::Expression, visit(ctx->expr()));
    auto ret = make_shared<ast::Return>(expr, token);

    return to_node(ret);
}

std::any Builder::visitPower(FusionParser::PowerContext* ctx) {
    Token* token = ctx->CARET()->getSymbol();
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(ast::BinaryOpType::POW, lhs,
                                                  rhs, token);

    return to_node(binop);
}

std::any Builder::visitMulDivMod(FusionParser::MulDivModContext* ctx) {
    Token* token;
    ast::BinaryOpType type;

    if (ctx->STAR() != nullptr) {
        type = ast::BinaryOpType::MUL;
        token = ctx->STAR()->getSymbol();
    } else if (ctx->SLASH() != nullptr) {
        type = ast::BinaryOpType::DIV;
        token = ctx->SLASH()->getSymbol();
    } else if (ctx->MOD() != nullptr) {
        type = ast::BinaryOpType::MOD;
        token = ctx->MOD()->getSymbol();
    } else {
        throw std::runtime_error(
            "Unrecognized operator when visiting mul div mod");
    }
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(type, lhs, rhs, token);

    return to_node(binop);
}

std::any Builder::visitAddSub(FusionParser::AddSubContext* ctx) {
    Token* token;
    ast::BinaryOpType type;

    if (ctx->PLUS() != nullptr) {
        type = ast::BinaryOpType::ADD;
        token = ctx->PLUS()->getSymbol();
    } else if (ctx->MINUS() != nullptr) {
        type = ast::BinaryOpType::SUB;
        token = ctx->MINUS()->getSymbol();
    } else {
        throw std::runtime_error("Unrecognized operator when visiting add sub");
    }
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(type, lhs, rhs, token);

    return to_node(binop);
}

std::any Builder::visitGtLtCond(FusionParser::GtLtCondContext* ctx) {
    Token* token;
    ast::BinaryOpType type;

    if (ctx->GT() != nullptr) {
        type = ast::BinaryOpType::GT;
        token = ctx->GT()->getSymbol();
    } else if (ctx->GE() != nullptr) {
        type = ast::BinaryOpType::GTE;
        token = ctx->GE()->getSymbol();
    } else if (ctx->LT() != nullptr) {
        type = ast::BinaryOpType::LT;
        token = ctx->LT()->getSymbol();
    } else if (ctx->LE() != nullptr) {
        type = ast::BinaryOpType::LTE;
        token = ctx->LE()->getSymbol();
    } else {
        throw std::runtime_error("Unrecognized operator when visiting gt lt");
    }
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(type, lhs, rhs, token);

    return to_node(binop);
}

std::any Builder::visitEqNeCond(FusionParser::EqNeCondContext* ctx) {
    Token* token;
    ast::BinaryOpType type;

    if (ctx->EQ() != nullptr) {
        type = ast::BinaryOpType::EQ;
        token = ctx->EQ()->getSymbol();
    } else if (ctx->NE() != nullptr) {
        type = ast::BinaryOpType::NE;
        token = ctx->NE()->getSymbol();
    } else {
        throw std::runtime_error("Unrecognized operator when visiting eq ne");
    }
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(type, lhs, rhs, token);

    return to_node(binop);
}

std::any Builder::visitAndOrCond(FusionParser::AndOrCondContext* ctx) {
    Token* token;
    ast::BinaryOpType type;

    if (ctx->DOR() != nullptr) {
        type = ast::BinaryOpType::OR;
        token = ctx->DOR()->getSymbol();
    } else if (ctx->DAND() != nullptr) {
        type = ast::BinaryOpType::AND;
        token = ctx->DAND()->getSymbol();
    } else {
        throw std::runtime_error("Unrecognized operator when visiting eq ne");
    }
    auto lhs = cast_node(ast::Expression, visit(ctx->expr()[0]));
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()[1]));
    auto binop = make_shared<ast::BinaryOperator>(type, lhs, rhs, token);

    return to_node(binop);
}

std::any Builder::visitUnary(FusionParser::UnaryContext* ctx) {
    Token* token;
    ast::UnaryOpType type;

    if (ctx->MINUS() != nullptr) {
        type = ast::UnaryOpType::MINUS;
        token = ctx->MINUS()->getSymbol();
    } else if (ctx->BANG() != nullptr) {
        type = ast::UnaryOpType::NOT;
        token = ctx->BANG()->getSymbol();
    } else {
        throw std::runtime_error("Unrecognized unary operator");
    }
    auto rhs = cast_node(ast::Expression, visit(ctx->expr()));
    auto binop = make_shared<ast::UnaryOperator>(type, rhs, token);

    return to_node(binop);
}

std::any Builder::visitAssignment(FusionParser::AssignmentContext* ctx) {
    Token* token = ctx->ID()->getSymbol();
    std::string name = ctx->ID()->getText();
    TypePtr type = make_shared<Unset>();

    auto var =
        make_shared<ast::Variable>(ast::Qualifier::Let, type, name, token);
    auto expr = cast_node(ast::Expression, visit(ctx->expr()));

    auto assn = make_shared<ast::Assignment>(var, expr, token);
    return to_node(assn);
}
std::any Builder::visitIf(FusionParser::IfContext* ctx) {
    Token* token = ctx->IF()->getSymbol();

    auto condition = cast_node(ast::Expression, visit(ctx->expr()));
    auto block = cast_node(ast::Block, visit(ctx->block()));

    std::optional<shared_ptr<ast::Conditional>> else_if = std::nullopt;
    if (ctx->else_() != nullptr) {
        else_if = cast_node(ast::Conditional, visit(ctx->else_()));
    }

    auto node = make_shared<ast::Conditional>(condition, block, else_if, token);
    return to_node(node);
}

std::any Builder::visitElse(FusionParser::ElseContext* ctx) {
    Token* token = ctx->ELSE()->getSymbol();

    if (ctx->if_() != nullptr) {
        return cast_node(ast::Node, visit(ctx->if_()));
    }

    auto block = cast_node(ast::Block, visit(ctx->block()));
    auto condition = make_shared<ast::BooleanLiteral>(true, token);
    auto node = make_shared<ast::Conditional>(condition, block, token);
    return to_node(node);
}

std::any Builder::visitLoop(FusionParser::LoopContext* ctx) {
    Token* token = ctx->FOR()->getSymbol();

    auto variable = cast_node(ast::Declaration, visit(ctx->declaration()));
    auto condition = cast_node(ast::Expression, visit(ctx->expr()));
    auto assignment = cast_node(ast::Assignment, visit(ctx->assignment()));
    auto body = cast_node(ast::Block, visit(ctx->block()));

    auto node =
        make_shared<ast::Loop>(variable, condition, assignment, body, token);
    return to_node(node);
}
