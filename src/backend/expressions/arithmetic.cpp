#include "backend/expressions/arithmetic.h"
#include "ast/ast.h"
#include "backend/types/boolean.h"
#include "backend/types/integer.h"
#include "shared/context.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

mlir::Value arithmetic::and_(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::AndOp>(lhs, rhs, ctx::t_bool);
}

mlir::Value arithmetic::or_(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binop<mlir::LLVM::OrOp>(lhs, rhs, ctx::t_bool);
}

mlir::Value arithmetic::pow(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    mlir::Value flhs = ctx::builder->create<mlir::LLVM::SIToFPOp>(
        *ctx::loc, ctx::f32->get_mlir(), lhs);

    mlir::Value pow = ctx::builder->create<mlir::LLVM::PowIOp>(
        *ctx::loc, ctx::builder->getF64Type(), flhs, rhs);
    return ctx::builder->create<mlir::LLVM::FPToSIOp>(
        *ctx::loc, ctx::i32->get_mlir(), pow);
}

mlir::Value arithmetic::eq(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::eq);
}

mlir::Value arithmetic::ne(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::ne);
}

mlir::Value arithmetic::gt(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::sgt);
}

mlir::Value arithmetic::gte(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::sge);
}

mlir::Value arithmetic::lt(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::slt);
}

mlir::Value arithmetic::lte(mlir::Value lhs, mlir::Value rhs, TypePtr type) {
    return binary_equality(lhs, rhs, type, mlir::LLVM::ICmpPredicate::sle);
}

mlir::Value arithmetic::not_(mlir::Value value) {
    mlir::Value f = boolean::create_bool(false);
    return eq(f, value, ctx::t_bool);
}

mlir::Value arithmetic::negate(mlir::Value value, TypePtr type) {
    mlir::Value neg_one = integer::create_i32(-1);
    return mul(neg_one, value, type);
}

mlir::Value arithmetic::binary_equality(mlir::Value lhs,
                                        mlir::Value rhs,
                                        TypePtr type,
                                        mlir::LLVM::ICmpPredicate predicate) {
    mlir::Value value = ctx::builder->create<mlir::LLVM::ICmpOp>(
        *ctx::loc, ctx::t_bool->get_mlir(), predicate, lhs, rhs);

    return value;
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
            return gt(lhs, rhs, type);
        case ast::BinaryOpType::GTE:
            return gte(lhs, rhs, type);
        case ast::BinaryOpType::LT:
            return lt(lhs, rhs, type);
        case ast::BinaryOpType::LTE:
            return lte(lhs, rhs, type);
        case ast::BinaryOpType::EQ:
            return eq(lhs, rhs, type);
        case ast::BinaryOpType::NE:
            return ne(lhs, rhs, type);
        case ast::BinaryOpType::AND:
            return and_(lhs, rhs, type);
        case ast::BinaryOpType::OR:
            return or_(lhs, rhs, type);
    }
}

mlir::Value arithmetic::unary_operation(mlir::Value rhs,
                                        ast::UnaryOpType op_type,
                                        TypePtr type) {
    switch (op_type) {
        case ast::UnaryOpType::MINUS:
            return negate(rhs, type);
        case ast::UnaryOpType::NOT:
            return not_(rhs);
    }
}
