#include "backend/expressions/flow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "shared/context.h"

void flow::branch(mlir::Value condition,
                  mlir::Block* b_true,
                  mlir::Block* b_false) {
    ctx::builder->create<mlir::LLVM::CondBrOp>(*ctx::loc, condition, b_true,
                                               b_false);
}

void flow::jump(mlir::Block* block) {
    ctx::builder->create<mlir::LLVM::BrOp>(*ctx::loc, block);
}
