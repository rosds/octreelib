cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OCTREELIB)


# ---[ Release/Debug specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG")
endif()

### ---[ Find universal dependencies
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


# check for SSE flags
include(${PROJECT_SOURCE_DIR}/cmake/octreelib_find_sse.cmake)
OCTREELIB_CHECK_FOR_SSE()

add_definitions( ${SSE_FLAGS} )


# Boost (required)
include(${PROJECT_SOURCE_DIR}/cmake/octreelib_find_boost.cmake)


# Eigen (required)
find_package(Eigen REQUIRED)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
add_definitions(-DEIGEN_USE_NEW_STDVECTOR
                -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)


# PCL (required)
find_package(PCL 1.5 REQUIRED COMPONENTS common)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



include_directories( ${PROJECT_SOURCE_DIR}/include )

set(SOURCES
        src/octreelib/spatialaggregate/octree.cpp
        src/octreelib/algorithm/downsample.cpp
        src/octreelib/feature/normalestimation.cpp
)

add_library(octreelib SHARED ${SOURCES})

##
#   Check if the compiler admits the C++11 standard 
##
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set_property(TARGET octreelib APPEND_STRING PROPERTY COMPILE_FLAGS 
        "-std=c++11 ")
elseif(COMPILER_SUPPORTS_CXX0X)
    set_property(TARGET octreelib APPEND_STRING PROPERTY 
        COMPILE_FLAGS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 
    support. Please use a different C++ compiler.")
endif() 


# configure install target
set(CMAKE_INSTALL_PREFIX "/usr/local")

install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/algorithm/downsample.h"
	DESTINATION include/octreelib/algorithm)
install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/algorithm/impl/downsample.hpp"
        DESTINATION include/octreelib/algorithm/impl)

install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/feature/normalestimation.h"
        DESTINATION include/octreelib/feature)
install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/feature/impl/normalestimation.hpp"
        DESTINATION include/octreelib/feature/impl)

install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/spatialaggregate/octree.h"
        DESTINATION include/octreelib/spatialaggregate)
install(FILES "${PROJECT_SOURCE_DIR}/include/octreelib/spatialaggregate/impl/octree.hpp"
        DESTINATION include/octreelib/spatialaggregate/impl)

install(FILES "${LIBRARY_OUTPUT_PATH}/liboctreelib.so"
	DESTINATION lib)

install(FILES "${PROJECT_SOURCE_DIR}/cmake/Modules/FindOctreeLib.cmake"
	DESTINATION /usr/share/cmake-2.8/Modules)


# configure uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Generate documentation (Optional)
find_package (Doxygen)
if (DOXYGEN_FOUND)
    # Fetch Doxygen configuration file
    configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

    # add 'make doc' directive
    add_custom_target (doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
        COMMENT "Generating API documentation" VERBATIM)
endif (DOXYGEN_FOUND)
