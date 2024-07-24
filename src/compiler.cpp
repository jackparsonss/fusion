#include <cassert>

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "FusionLexer.h"
#include "FusionParser.h"
#include "ast/passes/pass.h"
#include "compiler.h"
#include "errors/errors.h"
#include "errors/syntax.h"

Unit::Unit(std::string filename,
           LexerErrorListener* lexer_error,
           SyntaxErrorListener* syntax_error) {
    file = new antlr4::ANTLRFileStream();
    file->loadFromFile(filename);

    lexer = new fusion::FusionLexer(file);
    lexer->removeErrorListeners();
    lexer->addErrorListener(lexer_error);

    tokens = new antlr4::CommonTokenStream(lexer);

    parser = new fusion::FusionParser(tokens);
    parser->removeErrorListeners();
    parser->addErrorListener(syntax_error);

    tree = parser->file();
}

Unit::~Unit() {
    delete file;
    delete lexer;
    delete tokens;
    delete parser;
}

Compiler::Compiler(std::vector<std::string> filenames,
                   shared_ptr<SymbolTable> symbol_table,
                   unique_ptr<Backend> backend,
                   unique_ptr<Builder> builder) {
    this->symbol_table = symbol_table;
    this->backend = std::move(backend);
    this->builder = std::move(builder);

    lexer_error = new LexerErrorListener();
    syntax_error = new SyntaxErrorListener();

    for (const auto& filename : filenames) {
        auto unit = make_shared<Unit>(filename, lexer_error, syntax_error);
        this->units.push_back(unit);
    }
}

Compiler::~Compiler() {
    delete lexer_error;
    delete syntax_error;
}

void Compiler::build_ast() {
    for (const auto& unit : units) {
        try {
            builder->visit(unit->tree);
            assert(builder->has_ast());
        } catch (CompileTimeException const& e) {
            std::cerr << e.what() << std::endl;
            exit(1);
        }
    }
}

void Compiler::run_passes() {
    try {
        Pass::run_passes(builder->get_ast(), symbol_table);
        assert(builder->has_ast());
    } catch (CompileTimeException const& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void Compiler::xml() {
    builder->get_ast()->xml(0);
}

void Compiler::build_backend() {
    try {
        backend->traverse(this->builder->get_ast());
    } catch (CompileTimeException const& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void Compiler::to_object(std::string filename) {
    backend->to_object(filename + ".o");

    std::string command = "clang " + filename + ".o -o " + filename;
    system(command.c_str());

    command = "rm " + filename + ".o";
    system(command.c_str());
}

void Compiler::codegen(std::ofstream& outfile) {
    backend->codegen(outfile);
}
