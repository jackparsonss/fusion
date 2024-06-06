#include "backend/backend.h"

Backend::Backend(std::shared_ptr<ast::Block> ast) : BackendVisitor(ast) {
    ctx::builder->setInsertionPointToStart(ctx::module->getBody());
}

std::shared_ptr<ast::Block> Backend::traverse() {
    // temporarily set up main since llvm requires it, eventually all functions
    // will be handled together
    auto t_main = mlir::LLVM::LLVMFunctionType::get(ctx::i32, {}, false);
    mlir::LLVM::LLVMFuncOp main =
        ctx::builder->create<mlir::LLVM::LLVMFuncOp>(*ctx::loc, "main", t_main);
    mlir::Block* entry = main.addEntryBlock();
    ctx::builder->setInsertionPointToStart(entry);

    visit(ast);

    mlir::Value zero = integer::create_i32(0);
    io::print(zero);

    ctx::builder->create<mlir::LLVM::ReturnOp>(*ctx::loc, zero);

    if (mlir::failed(mlir::verify(*ctx::module))) {
        ctx::module->emitError("module failed to verify");
        return nullptr;
    }

    return ast;
}

void Backend::to_object(std::string filename) {
    mlir::registerLLVMDialectTranslation(ctx::context);

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module =
        mlir::translateModuleToLLVMIR(*ctx::module, llvm_context);

    if (!llvm_module) {
        std::cerr << "Failed to lower MLIR to LLVM" << std::endl;
        return;
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
        return;
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
        llvm::createLoopSimplifyCFGPass(),
        llvm::createInstructionCombiningPass(),
        llvm::createDeadCodeEliminationPass(),
        llvm::createDeadStoreEliminationPass(),
    };

    for (llvm::Pass* pass : optimizations) {
        pm.add(pass);
    }

    auto file_type = llvm::CodeGenFileType::CGFT_ObjectFile;
    if (target_machine->addPassesToEmitFile(pm, dest, nullptr, file_type)) {
        llvm::errs() << "target_machine can't emit a file of this type";
        return;
    }

    pm.run(*llvm_module);
    dest.flush();
}

void Backend::codegen(std::ostream& outstream) {
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
