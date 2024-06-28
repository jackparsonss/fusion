#pragma once

#include "BaseErrorListener.h"
#include "antlr4-runtime.h"

#define _NC "\033[0m"
#define _RED "\033[0;31m"
#define _BLUE "\033[0;34m"

class LexerErrorListener : public antlr4::BaseErrorListener {
   public:
    void syntaxError(antlr4::Recognizer* recognizer,
                     antlr4::Token* offending_symbol,
                     size_t line,
                     size_t char_position_in_line,
                     const std::string& msg,
                     std::exception_ptr e) override;
};

class SyntaxErrorListener : public antlr4::BaseErrorListener {
   public:
    void syntaxError(antlr4::Recognizer* recognizer,
                     antlr4::Token* offending_symbol,
                     size_t line,
                     size_t char_position_in_line,
                     const std::string& msg,
                     std::exception_ptr e) override;
};

void underline_error(antlr4::Recognizer* recognizer,
                     antlr4::Token* offending_symbol,
                     size_t line,
                     size_t char_position_in_line,
                     std::ostream& out);

void show_rule_stack(antlr4::Recognizer* recognizer, std::ostream& out);
