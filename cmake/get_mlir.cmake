find_package(MLIR REQUIRED CONFIG)

# Status messages about LLVM found.
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using Config.cmake in: ${MLIR_DIR}")

# Add mlir specific pieces to our build.
include_directories("${MLIR_INCLUDE_DIRS}")
include_directories("${LLVM_INCLUDE_DIRS}")
add_definitions("${MLIR_DEFINITIONS}")
