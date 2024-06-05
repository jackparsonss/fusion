cmake_minimum_required(VERSION 3.5)
project(FusionBase)

include(ExternalProject)
find_package(Git REQUIRED)
find_package(Java COMPONENTS Runtime REQUIRED) 

include("${CMAKE_SOURCE_DIR}/cmake/get_mlir.cmake")

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include("${CMAKE_SOURCE_DIR}/cmake/symlink_to_bin.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/get_antlr.cmake")

set(GRAMMAR_NAME "Fusion")
set(ANTLR_NAMESPACE "fusion")

include("${CMAKE_SOURCE_DIR}/cmake/antlr_generate.cmake")
include_directories("${CMAKE_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/runtime")
