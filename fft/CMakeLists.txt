# add the utils library
add_subdirectory(utils)
add_subdirectory(cpu)
add_subdirectory(gpu)

# set the list of public headers
FILE(GLOB HEADERS "Tester*.hh")

# add the library
add_library(ffttester SHARED TesterFactory.cc)

# add public headers for the library
set_target_properties(ffttester PROPERTIES PUBLIC_HEADER "${HEADERS}")

# add compile options for the library
target_compile_options(ffttester PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror>
)

# add the executable
add_executable(ffttest tester.cc)

# add compile options for the executable
target_compile_options(ffttest PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror>
)

# create a header file with the config information
configure_file(${CMAKE_SOURCE_DIR}/cmake/config.hh.in ${CMAKE_CURRENT_BINARY_DIR}/config.hh)

# add target includes for the library
target_include_directories(ffttester PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fft>
)

# add target includes for the executable
target_include_directories(ffttest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fft>
)

# add lib links for the library target
target_link_libraries(ffttester PUBLIC
    fftcpu
    fftgpu
)

# add lib links for the executable target
target_link_libraries(ffttest PUBLIC
    fftutils
    fftcpu
    fftgpu
    ffttester
)

# add install for the library target
install(TARGETS ffttester
    EXPORT "${FFT_EXPORT_TARGET}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fft"
)

# add install for the executable target
install(TARGETS ffttest
    EXPORT "${FFT_EXPORT_TARGET}"
)

