if(NOT EXISTS "${ANTLR_JAR}")
  message(FATAL_ERROR "Unable to find antlr jar. Did we miss a build step?")
endif()

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/grammar/${GRAMMAR_NAME}.g4" GRAMMAR_PATH)
set(GRAMMAR_PATH "${GRAMMAR_PATH}" CACHE FILEPATH "Path to the grammar file.")

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/gen/${ANTLR_NAMESPACE}" ANTLR_GEN_DIR)
set(ANTLR_GEN_DIR ${ANTLR_GEN_DIR} CACHE PATH "Generated source directory (ANTLR).")
file(TO_NATIVE_PATH "${ANTLR_GEN_DIR}" GEN_DIR_NATIVE)
message(STATUS "Generated source destination: ${GEN_DIR_NATIVE}")

if(DEFINED ANTLR_NAMESPACE)
  set(
    ANTLR_COMMAND
      "${Java_JAVA_EXECUTABLE}" -jar "${ANTLR_JAR}" -Werror -Dlanguage=Cpp -visitor
      -o "${ANTLR_GEN_DIR}" -package "${ANTLR_NAMESPACE}" "${GRAMMAR_PATH}"
  )
else()
  set(
    ANTLR_COMMAND
      "${Java_JAVA_EXECUTABLE}" -jar "${ANTLR_JAR}" -Werror -Dlanguage=Cpp -visitor
      -o "${ANTLR_GEN_DIR}" "${GRAMMAR_PATH}"
  )
endif()

# The list of our generated source files, necessary to specify what the output of the ANTLR
# generation command is so that the build system can propagate changes to the grammar appropriately.
set(
  ANTLR_GEN_SRC
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}BaseVisitor.cpp"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Lexer.cpp"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Parser.cpp"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Visitor.cpp"
)

set(
  ANTLR_GEN_HEADERS
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}BaseVisitor.h"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Lexer.h"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Parser.h"
    "${ANTLR_GEN_DIR}/${GRAMMAR_NAME}Visitor.h"
)

# Create the command to generate the sources.
add_custom_command(
  OUTPUT ${ANTLR_GEN_SRC} ${ANTLR_GEN_HEADERS}
  DEPENDS antlr "${GRAMMAR_PATH}"
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${ANTLR_GEN_DIR}
  COMMAND
    ${ANTLR_COMMAND}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Build a library from the generated sources.
add_library(parser STATIC ${ANTLR_GEN_SRC})
target_include_directories(parser PUBLIC ${ANTLR_INCLUDE_DIRS})
target_link_libraries(parser PUBLIC Threads::Threads)

