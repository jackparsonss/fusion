function(symlink_to_bin target)
  if(WIN32)
    message(WARN "CMake does not support generating symlinks on Windows.")
  else()
    message(STATUS "Generating custom command for symlinking ${target}.")
    add_custom_target(
      "symlink_${target}" ALL
      DEPENDS ${target}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/bin"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${target}>"
        "${CMAKE_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:${target}>"
      COMMENT
        "Symlinking target ${target} to ${CMAKE_SOURCE_DIR}/bin"
    )
  endif()
endfunction(symlink_to_bin)
