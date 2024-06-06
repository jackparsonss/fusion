#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <memory>

namespace ctx {
extern std::unique_ptr<mlir::Location> loc;
extern std::unique_ptr<mlir::ModuleOp> module;
extern std::shared_ptr<mlir::OpBuilder> builder;
extern mlir::MLIRContext context;

extern mlir::Type t_int;

extern void initialize_context();
}  // namespace ctx
