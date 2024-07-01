set(
  fusion_src_files
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/backend.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/visitor.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/io.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/utils.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/types/integer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/types/character.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/types/boolean.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/builtin/builtin.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/builtin/print.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/expressions/arithmetic.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/builder.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/symbol/scope.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/symbol/symbol.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/symbol/symbol_table.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/symbol/function_symbol.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/ast.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/passes/pass.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/passes/def_ref.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/passes/type_check.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/passes/builtin.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/type.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/character.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/integer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/float.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/boolean.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/type/type.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared/context.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/errors/syntax.cpp"
)

# Build our executable from the source files.
add_executable(fuse ${fusion_src_files})
target_include_directories(fuse PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(fuse antlr)

set(LLVM_LINK_COMPONENTS Core Support)
llvm_map_components_to_libnames(llvm_libs core)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

execute_process(
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-config --ldflags
    OUTPUT_VARIABLE LLVM_LD_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_SYSTEM_PROCESSOR "${LLVM_SYSTEM_LIBS}")
foreach(target ${LLVM_TARGETS_TO_BUILD})
  list(APPEND targets "LLVM${target}CodeGen")
endforeach()

# Add the antlr runtime and parser as libraries to link.
target_link_libraries(fuse PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
    ${targets}
)

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("fuse")
