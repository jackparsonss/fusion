#pragma once

#include "ast/ast.h"
#include "shared/type.h"

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

mlir::Value binary_equality(mlir::Value lhs,
                            mlir::Value rhs,
                            TypePtr type,
                            mlir::LLVM::ICmpPredicate predicate);
mlir::Value binary_operation(mlir::Value lhs,
                             mlir::Value rhs,
                             ast::BinaryOpType op_type,
                             TypePtr type);
}  // namespace arithmetic
