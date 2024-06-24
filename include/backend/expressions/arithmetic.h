#pragma once

#include "ast/ast.h"
#include "mlir/IR/Value.h"
#include "shared/type.h"

namespace arithmetic {
mlir::Value add(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value sub(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value mul(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value div(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value mod(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value pow(mlir::Value lhs, mlir::Value rhs, TypePtr type);
mlir::Value binary_operation(mlir::Value lhs,
                             mlir::Value rhs,
                             ast::BinaryOpType op_type,
                             TypePtr type);
}  // namespace arithmetic
