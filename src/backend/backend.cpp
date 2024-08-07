#include "backend/backend.h"
#include "backend/builtin/builtin.h"
#include "errors/errors.h"
#include "shared/context.h"

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
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

Backend::Backend() {
    ctx::builder->setInsertionPointToStart(ctx::module->getBody());
    builtin::define_all();
}

shared_ptr<ast::Block> Backend::traverse(shared_ptr<ast::Block> ast) {
    visit(ast);

    if (mlir::failed(mlir::verify(*ctx::module))) {
        throw BackendError("backend failed to build");
    }

    return ast;
}

void Backend::to_object(std::string filename) {
    mlir::registerBuiltinDialectTranslation(ctx::context);
    mlir::registerLLVMDialectTranslation(ctx::context);

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module =
        mlir::translateModuleToLLVMIR(*ctx::module, llvm_context);

    if (!llvm_module) {
        std::cerr << "Failed to lower MLIR to LLVM" << std::endl;
        exit(1);
    }
    llvm::verifyModule(*llvm_module, &llvm::errs());

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    std::string target_triple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (!target) {
        llvm::errs() << error;
        exit(1);
    }

    std::string CPU = "generic";
    llvm::TargetMachine* target_machine = target->createTargetMachine(
        target_triple, CPU, "", {}, llvm::Reloc::PIC_);

    llvm_module->setDataLayout(target_machine->createDataLayout());
    llvm_module->setTargetTriple(target_triple);

    std::error_code EC;
    llvm::raw_fd_ostream dest(filename, EC, llvm::sys::fs::OF_None);
    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message();
    }

    llvm::legacy::PassManager pm;
    std::vector<llvm::Pass*> optimizations = {
        llvm::createInstructionCombiningPass(),
        llvm::createDeadCodeEliminationPass(),
    };

    for (llvm::Pass* pass : optimizations) {
        pm.add(pass);
    }

    auto file_type = llvm::CodeGenFileType::ObjectFile;
    if (target_machine->addPassesToEmitFile(pm, dest, nullptr, file_type)) {
        llvm::errs() << "target_machine can't emit a file of this type";
        return;
    }

    pm.run(*llvm_module);
    dest.flush();
}

void Backend::codegen(std::ostream& outstream) {
    mlir::registerBuiltinDialectTranslation(ctx::context);
    mlir::registerLLVMDialectTranslation(ctx::context);

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module =
        mlir::translateModuleToLLVMIR(*ctx::module, llvm_context);

    if (!llvm_module) {
        std::cerr << "Failed to translate to LLVM IR" << std::endl;
        return;
    }

    llvm::verifyModule(*llvm_module, &llvm::errs());

    // print LLVM IR to file
    llvm::raw_os_ostream output(outstream);
    output << *llvm_module;
}
