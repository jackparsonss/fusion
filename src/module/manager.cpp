#include "module/manager.h"
#include <string>
#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "FusionLexer.h"
#include "FusionParser.h"
#include "ast/ast.h"
#include "errors/syntax.h"

module::Unit::Unit(std::string filename) {
    auto lexer_error = new LexerErrorListener();
    auto syntax_error = new SyntaxErrorListener();

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

module::Unit::~Unit() {
    delete file;
    delete lexer;
    delete tokens;
    delete parser;
}

shared_ptr<module::Unit> module::Manager::compile_unit(std::string module) {
    auto m = module_cache.find(module);
    if (m != module_cache.end()) {
        return m->second;
    }

    auto unit = make_shared<module::Unit>(module + ".fuse");
    module_cache[module] = unit;

    return unit;
}
