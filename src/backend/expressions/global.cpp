#include "backend/expressions/global.h"
#include "ast/ast.h"
#include "backend/utils.h"
#include "mlir/IR/Types.h"

void GlobalManager::define(std::string name, shared_ptr<ast::Expression> expr) {
    mlir::Type type = expr->get_type()->get_mlir();
    utils::define_global(type, name);

    Global gl{expr, name};
    store.push_back(gl);
    globals[name] = store.size() - 1;
}

Global GlobalManager::resolve(std::string name) {
    if (!exists(name)) {
        throw std::runtime_error("Global Manager found undefined global: " +
                                 name);
    }

    unsigned int index = globals[name];
    return store[index];
}

bool GlobalManager::exists(std::string name) {
    return globals.contains(name);
}

mlir::Value GlobalManager::value(std::string name, TypePtr type) {
    mlir::Value address = utils::get_global_address(name);
    return utils::load(address, type);
}

std::vector<Global> GlobalManager::get_store() {
    return this->store;
}
