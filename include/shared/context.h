#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <memory>

#include "shared/type.h"

using std::shared_ptr, std::make_shared, std::unique_ptr;

namespace ctx {
extern unique_ptr<mlir::Location> loc;
extern unique_ptr<mlir::ModuleOp> module;
extern shared_ptr<mlir::OpBuilder> builder;
extern mlir::MLIRContext context;

extern shared_ptr<Type> ch;
extern shared_ptr<Type> i32;
extern shared_ptr<Type> f32;
extern shared_ptr<Type> none;
extern shared_ptr<Type> t_bool;

extern void initialize_context();
}  // namespace ctx
