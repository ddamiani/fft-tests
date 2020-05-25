FILE(GLOB SOURCES "*.cc")
FILE(GLOB HEADERS "*.hh")

# add the library
add_library(fftcpu SHARED ${SOURCES})

# add public headers for the library
set_target_properties(fftcpu PROPERTIES PUBLIC_HEADER "${HEADERS}")

# add compile options for the library
target_compile_options(fftcpu PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror>
)

# add target includes for the library
target_include_directories(fftcpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fft/cpu>
)

# add link to fftw3 and fftutils
target_link_libraries(fftcpu
    PRIVATE fftutils
    PUBLIC FFTW::fftw3
)

install(TARGETS fftcpu
    EXPORT "${FFT_EXPORT_TARGET}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fft/cpu"
)

# install the fttw dlls on windows
if(WIN32)
    install(FILES ${FFTW_DLL_FILES}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()
