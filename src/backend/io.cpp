#include "backend/io.h"
#include "backend/types/integer.h"
#include "backend/utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "shared/context.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace {
void print(mlir::Value value, TypePtr type, std::string name) {
    mlir::Value global_ptr = utils::get_global_address(name);
    auto gtype = mlir::LLVM::LLVMArrayType::get(ctx::ch->get_mlir(), 3);

    mlir::Value cst0 = integer::create_i32(0);
    mlir::Value arg = ctx::builder->create<mlir::LLVM::GEPOp>(
        *ctx::loc, ctx::ch->get_pointer(), gtype, global_ptr,
        mlir::ValueRange{cst0, cst0});

    mlir::LLVM::LLVMFuncOp func =
        ctx::module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    utils::call(func, {arg, value});
}
}  // namespace

void io::printf(mlir::Value value, TypePtr type) {
    print(value, type, type->get_name());
}

void io::println(mlir::Value value, TypePtr type) {
    print(value, type, "newline_" + type->get_name());
}
