#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "ast/ast.h"
#include "backend/visitor.h"

#include "mlir/IR/Value.h"

using std::shared_ptr;

class Backend : public BackendVisitor {
   private:
    std::unordered_map<std::string, mlir::Value> variables;

   public:
    explicit Backend(shared_ptr<ast::Block>);

    void codegen(std::ostream& outstream);
    void to_object(std::string filename);

    shared_ptr<ast::Block> traverse() override;
    mlir::Value visit_block(shared_ptr<ast::Block>) override;
    mlir::Value visit_integer_literal(shared_ptr<ast::IntegerLiteral>) override;
    mlir::Value visit_variable(shared_ptr<ast::Variable>) override;
    mlir::Value visit_declaration(shared_ptr<ast::Declaration>) override;
    mlir::Value visit_function(shared_ptr<ast::Function>) override;
    mlir::Value visit_call(shared_ptr<ast::Call>) override;
    mlir::Value visit_parameter(shared_ptr<ast::Parameter>) override;
    mlir::Value visit_return(shared_ptr<ast::Return>) override;
};
