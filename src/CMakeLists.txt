set(
  fusion_src_files
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/backend.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/ast_builder.cpp"
)

# Build our executable from the source files.
add_executable(fuse ${fusion_src_files})
target_include_directories(fuse PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(fuse antlr)

set(LLVM_LINK_COMPONENTS Core Support)
llvm_map_components_to_libnames(llvm_libs core)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Add the antlr runtime and parser as libraries to link.
target_link_libraries(fuse PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
)

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("fuse")
