# Finds the include directories
find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    PATH_SUFFIXES include
    HINTS ${FFTW_ROOT} env FFTW_ROOT
)
mark_as_advanced(FFTW_INCLUDE_DIRS)

# Finds the libraries (and dlls for windows)
foreach(libname IN ITEMS fftw3 fftw3f fftw3l)
    find_library(FFTW_${libname}
        NAMES ${libname} lib${libname}-3
        PATH_SUFFIXES lib
        HINTS ${FFTW_ROOT} env FFTW_ROOT
    )
    mark_as_advanced(FFTW_${libname}) 

    if(WIN32)
        find_file(FFTW_${libname}_DLL
            NAMES ${libname}.dll lib${libname}-3.dll
            PATH_SUFFIXES lib
            HINTS ${FFTW_ROOT} env FFTW_ROOT
        )
        list(APPEND FFTW_DLL_FILES "${FFTW_${libname}_DLL}")
        mark_as_advanced(FFTW_${libname}_DLL) 
    endif()
endforeach(libname)
message(STATUS "The values in " ${FFTW_DLL_FILES})

if(WIN32)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
        FFTW DEFAULT_MSG
        FFTW_INCLUDE_DIRS
        FFTW_fftw3
        FFTW_fftw3f
        FFTW_fftw3l
        FFTW_fftw3_DLL
        FFTW_fftw3f_DLL
        FFTW_fftw3l_DLL 
    )
else()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
        FFTW DEFAULT_MSG
        FFTW_INCLUDE_DIRS
        FFTW_fftw3
        FFTW_fftw3f
        FFTW_fftw3l
    )
endif()

if(FFTW_FOUND)
    set(FFTW_DLL_FILES "${FFTW_fftw3_DLL}" "${FFTW_fftw3f_DLL}" "${FFTW_fftw3l_DLL}")

    if(NOT TARGET FFTW::fftw3)
        add_library(FFTW::fftw3 UNKNOWN IMPORTED)
        set_target_properties(FFTW::fftw3 PROPERTIES
            IMPORTED_LOCATION ${FFTW_fftw3}
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET FFTW::fftw3f)
        add_library(FFTW::fftw3f UNKNOWN IMPORTED)
        set_target_properties(FFTW::fftw3f PROPERTIES
            IMPORTED_LOCATION ${FFTW_fftw3f}
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET FFTW::fftw3l)
        add_library(FFTW::fftw3l UNKNOWN IMPORTED)
        set_target_properties(FFTW::fftw3l PROPERTIES
            IMPORTED_LOCATION ${FFTW_fftw3l}
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
        )
    endif()
endif()

