#include "backend/expressions/arithmetic.h"
#include "ast/ast.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "shared/context.h"

namespace {
template <typename OpTy>
mlir::Value binop(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return ctx::builder->create<OpTy>(*ctx::loc, type->get_mlir(), lhs, rhs);
}

}  // namespace
//
mlir::Value arithmetic::add(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::AddOp>(lhs, rhs, type);
}

mlir::Value arithmetic::sub(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::SubOp>(lhs, rhs, type);
}

mlir::Value arithmetic::mul(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::MulOp>(lhs, rhs, type);
}

mlir::Value arithmetic::div(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::SDivOp>(lhs, rhs, type);
}

mlir::Value arithmetic::mod(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::SRemOp>(lhs, rhs, type);
}

mlir::Value arithmetic::pow(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    mlir::Value flhs = ctx::builder->create<mlir::LLVM::SIToFPOp>(
        *ctx::loc, ctx::f32->get_mlir(), lhs);

    mlir::Value pow = ctx::builder->create<mlir::LLVM::PowIOp>(
        *ctx::loc, ctx::builder->getF64Type(), flhs, rhs);
    return ctx::builder->create<mlir::LLVM::FPToSIOp>(
        *ctx::loc, ctx::i32->get_mlir(), pow);
}

mlir::Value arithmetic::binary_operation(mlir::Value lhs,
                                         mlir::Value rhs,
                                         ast::BinaryOpType op_type,
                                         TypePtr type) {
    switch (op_type) {
        case ast::BinaryOpType::ADD:
            return add(lhs, rhs, type);
        case ast::BinaryOpType::POW:
            return pow(lhs, rhs, type);
        case ast::BinaryOpType::SUB:
            return sub(lhs, rhs, type);
        case ast::BinaryOpType::MUL:
            return mul(lhs, rhs, type);
        case ast::BinaryOpType::DIV:
            return div(lhs, rhs, type);
        case ast::BinaryOpType::MOD:
            return mod(lhs, rhs, type);
        case ast::BinaryOpType::GT:
            throw std::runtime_error("GT not yet implemented on backend");
        case ast::BinaryOpType::GTE:
            throw std::runtime_error("GTE not yet implemented on backend");
        case ast::BinaryOpType::LT:
            throw std::runtime_error("LT not yet implemented on backend");
        case ast::BinaryOpType::LTE:
            throw std::runtime_error("LTE not yet implemented on backend");
        case ast::BinaryOpType::EQ:
            throw std::runtime_error("EQ not yet implemented on backend");
        case ast::BinaryOpType::NE:
            throw std::runtime_error("NE not yet implemented on backend");
            break;
    }
}
