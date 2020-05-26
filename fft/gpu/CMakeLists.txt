FILE(GLOB SOURCES "*.c[cu]")
FILE(GLOB HEADERS "*.hh")
# filter out the cuda headers
list(FILTER HEADERS EXCLUDE REGEX "Cuda.*.hh$")

# add the library
add_library(fftgpu SHARED ${SOURCES})

# add public headers for the library
set_target_properties(fftgpu PROPERTIES PUBLIC_HEADER "${HEADERS}")

# add compile options for the library
target_compile_options(fftgpu PRIVATE --Werror cross-execution-space-call,deprecated-declarations,reorder)

# add target includes for the library
target_include_directories(fftgpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fft/gpu>
)

# add link to cuFFT
target_link_libraries(fftgpu
    PRIVATE
    fftutils
    CUDA::cudart
    PUBLIC
    CUDA::cufft
)

install(TARGETS fftgpu
    EXPORT "${FFT_EXPORT_TARGET}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fft/gpu"
)
