#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <memory>
#include <stack>

#include "shared/type/type.h"

using std::shared_ptr, std::make_shared, std::unique_ptr;

namespace ctx {
extern unique_ptr<mlir::Location> loc;
extern unique_ptr<mlir::ModuleOp> module;
extern shared_ptr<mlir::OpBuilder> builder;
extern mlir::MLIRContext context;

extern TypePtr ch;
extern TypePtr any;
extern TypePtr i32;
extern TypePtr i64;
extern TypePtr f32;
extern TypePtr none;
extern TypePtr bool_;

extern std::vector<TypePtr> primitives;
extern std::stack<mlir::LLVM::LLVMFuncOp> function_stack;

extern void initialize_context();
extern mlir::LLVM::LLVMFuncOp current_function();
}  // namespace ctx
