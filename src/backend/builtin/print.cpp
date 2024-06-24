#include "backend/builtin/print.h"
#include "backend/io.h"
#include "shared/context.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"

namespace {
void create_type_str(TypePtr type) {
    std::string ty = type->get_specifier();
    auto gvalue = mlir::StringRef(ty.c_str(), 3);

    auto gtype =
        mlir::LLVM::LLVMArrayType::get(ctx::ch->get_mlir(), gvalue.size());

    ctx::builder->create<mlir::LLVM::GlobalOp>(
        *ctx::loc, gtype, true, mlir::LLVM::Linkage::Internal, type->get_name(),
        ctx::builder->getStringAttr(gvalue), 0);
}

void setupPrintf() {
    auto func_type = mlir::LLVM::LLVMFunctionType::get(
        ctx::i32->get_mlir(), ctx::ch->get_pointer(), true);

    ctx::builder->create<mlir::LLVM::LLVMFuncOp>(*ctx::loc, "printf",
                                                 func_type);
}

void newline_type_str(TypePtr type) {
    std::string ty = type->get_specifier() + "\n";
    auto gvalue = mlir::StringRef(ty.c_str(), 4);

    auto gtype =
        mlir::LLVM::LLVMArrayType::get(ctx::ch->get_mlir(), gvalue.size());

    ctx::builder->create<mlir::LLVM::GlobalOp>(
        *ctx::loc, gtype, true, mlir::LLVM::Linkage::Internal,
        "newline_" + type->get_name(), ctx::builder->getStringAttr(gvalue), 0);
}
}  // namespace

void builtin::define_all() {
    setupPrintf();

    create_type_str(ctx::i32);
    create_type_str(ctx::ch);

    newline_type_str(ctx::i32);
    newline_type_str(ctx::ch);

    define_print(ctx::i32);
    define_print(ctx::ch);

    define_println(ctx::i32);
    define_println(ctx::ch);
}

void builtin::define_print(TypePtr type) {
    auto func_type = mlir::LLVM::LLVMFunctionType::get(
        ctx::none->get_mlir(), {type->get_mlir()}, false);

    mlir::LLVM::LLVMFuncOp func = ctx::builder->create<mlir::LLVM::LLVMFuncOp>(
        *ctx::loc, "print_" + type->get_name(), func_type);

    mlir::Block* body = func.addEntryBlock();
    ctx::builder->setInsertionPointToStart(body);

    mlir::Value arg = func.getArgument(0);
    io::printf(arg, type);

    ctx::builder->create<mlir::LLVM::ReturnOp>(*ctx::loc, mlir::Value{});
    ctx::builder->setInsertionPointToEnd(ctx::module->getBody());
}

void builtin::define_println(TypePtr type) {
    auto func_type = mlir::LLVM::LLVMFunctionType::get(
        ctx::none->get_mlir(), {type->get_mlir()}, false);

    mlir::LLVM::LLVMFuncOp func = ctx::builder->create<mlir::LLVM::LLVMFuncOp>(
        *ctx::loc, "println_" + type->get_name(), func_type);

    mlir::Block* body = func.addEntryBlock();
    ctx::builder->setInsertionPointToStart(body);

    mlir::Value arg = func.getArgument(0);
    io::println(arg, type);

    ctx::builder->create<mlir::LLVM::ReturnOp>(*ctx::loc, mlir::Value{});
    ctx::builder->setInsertionPointToEnd(ctx::module->getBody());
}
