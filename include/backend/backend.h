#pragma once

#include <memory>
#include <ostream>

#include "ast/ast.h"
#include "backend/context.h"
#include "backend/io.h"
#include "backend/types/integer.h"
#include "backend/visitor.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

using std::shared_ptr;

class Backend : public BackendVisitor {
   public:
    explicit Backend(std::shared_ptr<ast::Block>);

    void codegen(std::ostream& outstream);
    void to_object(std::string filename);

    std::shared_ptr<ast::Block> traverse() override;
    mlir::Value visit_block(std::shared_ptr<ast::Block>) override;
    mlir::Value visit_integer_literal(
        shared_ptr<ast::IntegerLiteral> node) override;
};
