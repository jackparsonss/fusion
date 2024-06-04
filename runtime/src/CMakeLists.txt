# Gather our source files in this directory.
set(
  fusion_rt_files
  "${CMAKE_CURRENT_SOURCE_DIR}/io.c"
)

# Build our executable from the source files.
add_library(fusert SHARED ${fusion_rt_files})
target_include_directories(fuse PUBLIC ${RUNTIME_INCLUDE})

# Symbolic link our library to the base directory so we don't have to go searching for it.
symlink_to_bin("fusert")
