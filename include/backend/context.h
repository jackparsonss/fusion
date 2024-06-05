#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include "mlir/IR/Types.h"

namespace ctx {
extern mlir::Location loc;
extern mlir::ModuleOp module;
extern mlir::MLIRContext context;
extern std::shared_ptr<mlir::OpBuilder> builder;

extern mlir::Type t_int;

extern void initialize_context();
}  // namespace ctx
