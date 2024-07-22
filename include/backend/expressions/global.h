#pragma once

#include <unordered_map>

#include "ast/ast.h"

struct Global {
    shared_ptr<ast::Expression> node;
    std::string name;
};

class GlobalManager {
   private:
    std::vector<Global> store;
    std::unordered_map<std::string, unsigned int> globals;

   public:
    void define(std::string name, shared_ptr<ast::Expression> expr);
    Global resolve(std::string name);
    bool exists(std::string name);
    std::vector<Global> get_store();
};
