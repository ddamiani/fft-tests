cmake_minimum_required(VERSION 3.24)

# set the project name
set(PROJECT_NAME FastFourieTransformTests)
# set the project name
project(${PROJECT_NAME}
    VERSION 1.0
    DESCRIPTION "Package for testing various FFTs"
    LANGUAGES CXX CUDA
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check for the CUDA toolkit
find_package(CUDAToolkit REQUIRED)
# check for FFTW
find_package(FFTW REQUIRED)

# include GNUInstallDir defaults
include(GNUInstallDirs)
# set some variables used by the export
set(FFT_EXPORT_NAME "fftTests")
set(FFT_EXPORT_TARGET "${FFT_EXPORT_NAME}Targets")
set(FFT_EXPORT_CONFIG "${FFT_EXPORT_NAME}Config")
set(FFT_EXPORT_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${fftTests}")

# setup rpath for executables
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# have windows automatically export all symbols
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hh.in config.hh)

# add the fft directory
add_subdirectory(fft)

# include package helpers
include(CMakePackageConfigHelpers)

install(EXPORT "${FFT_EXPORT_TARGET}"
    FILE "${FFT_EXPORT_TARGET}.cmake"
    NAMESPACE "${FFT_EXPORT_NAME}::"
    DESTINATION "${FFT_EXPORT_DESTINATION}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${FFT_EXPORT_CONFIG}.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${FFT_EXPORT_CONFIG}.cmake"
    INSTALL_DESTINATION "${FFT_EXPORT_DESTINATION}"
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${FFT_EXPORT_CONFIG}.cmake"
    DESTINATION "${FFT_EXPORT_DESTINATION}"
)
