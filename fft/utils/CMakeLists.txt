FILE(GLOB SOURCES "*.cc")
FILE(GLOB HEADERS "*.hh")

# test
add_subdirectory(CLI11)

# add the library
add_library(fftutils SHARED ${SOURCES})

# add lib links for the library target
target_link_libraries(fftutils PRIVATE
    CLI11::CLI11
)

# add public headers for the library
set_target_properties(fftutils PROPERTIES PUBLIC_HEADER "${HEADERS}")

# add compile options for the library
target_compile_options(fftutils PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror>
)

# add target includes for the library
target_include_directories(fftutils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fft/utils>
)

install(TARGETS fftutils
    EXPORT "${FFT_EXPORT_TARGET}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fft/utils"
)
