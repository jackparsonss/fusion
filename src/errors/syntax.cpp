#include "errors/syntax.h"
#include "errors/errors.h"

void SyntaxErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                      antlr4::Token* offending_symbol,
                                      size_t line,
                                      size_t char_position_in_line,
                                      const std::string& msg,
                                      std::exception_ptr e) {
    std::ostringstream out;
    underline_error(recognizer, offending_symbol, line, char_position_in_line,
                    out);
    show_rule_stack(recognizer, out);
    throw SyntaxError(line, msg + out.rdbuf()->str());
}

void LexerErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                     antlr4::Token* offending_symbol,
                                     size_t line,
                                     size_t char_position_in_line,
                                     const std::string& msg,
                                     std::exception_ptr e) {
    throw SyntaxError(line, msg);
}

void underline_error(antlr4::Recognizer* recognizer,
                     antlr4::Token* offending_symbol,
                     size_t line,
                     size_t char_position_in_line,
                     std::ostream& out) {
    auto tokens =
        dynamic_cast<antlr4::CommonTokenStream*>(recognizer->getInputStream());
    auto input = tokens->getTokenSource()->getInputStream()->toString();
    std::string error_line = "\n";
    for (size_t i = 0, line_counter = 0;
         i < input.size() && line_counter <= line; i++) {
        if (input[i] == '\n') {
            line_counter++;
        } else if (line_counter == line - 1) {
            error_line += input[i];
        }
    }
    auto length = offending_symbol->getStopIndex() -
                  offending_symbol->getStartIndex() + 1;
    out << error_line << std::endl
        << _RED << std::string(char_position_in_line, ' ')
        << std::string(length, '^') << _NC << std::endl;
}

void show_rule_stack(antlr4::Recognizer* recognizer, std::ostream& out) {
    auto parser = dynamic_cast<antlr4::Parser*>(recognizer);
    auto rule_stack = parser->getRuleInvocationStack();
    out << "Rule stack:" << std::endl;
    size_t i = rule_stack.size();
    out << _BLUE << rule_stack[--i] << std::endl;
    for (size_t indent_level = 0; i-- > 0; indent_level++) {
        out << std::string(indent_level * 2, ' ') << "└─" << rule_stack[i]
            << std::endl;
    }
    out << _NC;
}
