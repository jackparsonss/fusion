#include <cassert>
#include <filesystem>

#include "ast/passes/pass.h"
#include "compiler.h"
#include "errors/errors.h"

Compiler::Compiler(fs::path entry,
                   shared_ptr<SymbolTable> symbol_table,
                   unique_ptr<Backend> backend,
                   unique_ptr<Builder> builder) {
    this->symbol_table = symbol_table;
    this->backend = std::move(backend);
    this->builder = std::move(builder);

    this->entry = make_shared<module::Unit>(entry);
}

Compiler::~Compiler() {
    delete lexer_error;
    delete syntax_error;
}

void Compiler::build_ast() {
    try {
        builder->visit(entry->tree);
        assert(builder->has_ast());
    } catch (CompileTimeException const& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
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
