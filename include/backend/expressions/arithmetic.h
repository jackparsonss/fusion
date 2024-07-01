#pragma once

#include "ast/ast.h"
#include "shared/type/type.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Value.h"

namespace arithmetic {
mlir::Value add(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value sub(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value mul(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value div(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value mod(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value pow(mlir::Value lhs, mlir::Value rhs, TypePtr type);

mlir::Value eq(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value ne(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value gt(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value gte(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value lt(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value lte(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value and_(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value or_(mlir::Value lhs, mlir::Value rhs, TypePtr type);

mlir::Value not_(mlir::Value value);
mlir::Value negate(mlir::Value value, TypePtr type);

mlir::Value binary_equality(mlir::Value lhs,
                            mlir::Value rhs,
                            TypePtr type,
                            mlir::LLVM::ICmpPredicate predicate);
mlir::Value binary_operation(mlir::Value lhs,
                             mlir::Value rhs,
                             ast::BinaryOpType op_type,
                             TypePtr type);
mlir::Value unary_operation(mlir::Value rhs,
                            ast::UnaryOpType op_type,
                            TypePtr type);
}  // namespace arithmetic
