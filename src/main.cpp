#include "ast/builder.h"
#include "ast/passes/pass.h"
#include "ast/symbol/symbol_table.h"
#include "backend/backend.h"
#include "compiler.h"
#include "module/manager.h"
#include "shared/context.h"

#include <iostream>
#include <memory>

using std::shared_ptr, std::unique_ptr, std::make_shared, std::make_unique;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "must provide an entry point file" << std::endl;
        exit(1);
    }

    ctx::initialize_context();
    shared_ptr<SymbolTable> symbol_table = make_shared<SymbolTable>();
    shared_ptr<module::Manager> module_manager = make_shared<module::Manager>();
    unique_ptr<Builder> builder =
        make_unique<Builder>(symbol_table, module_manager);
    unique_ptr<Backend> backend = make_unique<Backend>();

    std::string entry = std::string(argv[1]);
    Compiler compiler =
        Compiler(entry, symbol_table, std::move(backend), std::move(builder));

    compiler.build_ast();
    for (size_t i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--xml") {
            compiler.xml();
        }
    }

    compiler.run_passes();
    for (size_t i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--pass-xml") {
            compiler.xml();
        }
    }

    compiler.build_backend();
    for (size_t i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--emit-llvm" && i == argc - 1) {
            std::cerr << "You must provide a file emit llvm ir to" << std::endl;
            return 1;
        }

        if (arg == "-o" && i == argc - 1) {
            std::cerr << "You must provide a file to compile" << std::endl;
            return 1;
        }

        if (arg == "-o") {
            std::string filename = std::string(argv[i + 1]);
            compiler.to_object(filename);
        }

        if (arg == "--emit-llvm") {
            std::ofstream outfile(argv[i + 1]);
            compiler.codegen(outfile);
        }
    }

    return 0;
}
