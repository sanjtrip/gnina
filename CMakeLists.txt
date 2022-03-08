cmake_minimum_required(VERSION 3.12)
if (DEFINED USE_ROCM)
   project (gnina C CXX)
else()
   project (gnina C CXX CUDA)
endif()

set (CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

#disable parts of caffe we don't need
set(USE_LMDB 0)
set(USE_LEVELDB 0)
set(USE_OPENCV 0)

# Sanjay
#if(POLICY CMP0104)
#  cmake_policy(SET CMP0104 OLD) #TODO: clean up Cuda.cmake to be modern to avoid this workaround
#endif()

set(CMAKE_CXX_STANDARD 14)
if (DEFINED USE_ROCM)
   set(CMAKE_HIP_STANDARD 14)
else()
   set(CMAKE_CUDA_STANDARD 14)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# Sanjay
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unknown-pragmas  -Wno-unused-command-line-argument ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-uninitialized -MD -MT ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_DEVICE_COMPILE__ -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__ -std=c++17 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hip-link -fgpu-rdc -Wno-deprecated-register ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/root/scratch/gnina/include ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/root/scratch/gnina/caffe/src ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include/ -I/usr/local/include/libmolgrid -I/usr/local/include/openbabel3 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /opt/rocm-5.0.0/include -isystem /opt/rocm-5.0.0/hsa/include ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /opt/rocm-5.0.0/hip/include -isystem /opt/rocm-5.0.0/hiprand/include ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /opt/rocm-5.0.0/rocrand/include ")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (DEFINED USE_ROCM)
  find_package( ROCM REQUIRED CONFIG QUIET PATHS ${USE_ROCM} )
  if( NOT ROCM_FOUND )
    message(STATUS "ROCm not installed... done")
  else()
    message(STATUS "Found ROCm installed... done")
  endif() 
else()
   set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
   find_package(CUDA 9.0 REQUIRED)
endif()

if (NOT DEFINED USE_ROCM)
   if(NOT ${CUDA_cublas_device_LIBRARY}) #workaround for oldercmake/newer cuda issue
      set(CUDA_cublas_device_LIBRARY "")
   endif()
endif()

message(STATUS "Finding OpenBabel3 packages...")
find_package( OpenBabel3 REQUIRED CONFIG QUIET PATHS ${BABEL_PATH} )
#find_package(OpenBabel3 REQUIRED)
#set(OpenBabel3_FOUND TRUE)
if( NOT OpenBabel3_FOUND )
   message(STATUS "OpenBabel3 not installed... done")
else()
   message(STATUS "Found OpenBabel3 installed... done")
endif() 

include_directories(SYSTEM ${OPENBABEL3_INCLUDE_DIR})

#set(OpenBabel3_LIBRARIES $<TARGET_FILE:openbabel>)
#include_directories($<TARGET_PROPERTY:openbabel,INTERFACE_INCLUDE_DIRECTORIES>)

#print_target_properties(openbabel3)
#set(OpenBabel3_LIBRARIES $<TARGET_FILE:openbabel3>)
#include_directories($<TARGET_PROPERTY:openbabel3,INTERFACE_INCLUDE_DIRECTORIES>)
#include_directories(SYSTEM ${OPENBABEL3_INCLUDE_DIR})
#include_directories(${OpenBabel3_INCLUDE_DIRS})
  
find_package(ZLIB REQUIRED)

if (DEFINED USE_ROCM)
   include( ROCMSetupVersion )
   include( ROCMCreatePackage )
   include( ROCMInstallTargets )
   include( ROCMPackageConfigHelpers )
   include( ROCMInstallSymlinks )
endif()

find_package(libmolgrid)
if(NOT LIBMOLGRID_LIBRARY)
 message("libmolgrid will be fetched from git")
 include(ExternalProject)
 find_package(Git REQUIRED)
 set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
 ExternalProject_Add(libmolgrid
    GIT_REPOSITORY https://github.com/gnina/libmolgrid
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )
  set(LIBMOLGRID_INCLUDE_DIR ${EXTERNAL_INSTALL_LOCATION}/include)
  set(LIBMOLGRID_LIBRARY ${EXTERNAL_INSTALL_LOCATION}/lib/libmolgrid.a)
  include_directories(${LIBMOLGRID_INCLUDE_DIR})
else()
  add_custom_target(libmolgrid)
endif()


####### USE_ROCM
list( APPEND CMAKE_PREFIX_PATH ${USE_ROCM}/llvm ${USE_ROCM}/hip )

if(NOT SKIP_LIBRARY)
  message(STATUS "DEBUG-3 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  set( Tensile_ARCHITECTURE "all" CACHE STRING "Tensile to use which architecture?")
  # Sanjay
  #set_property( CACHE Tensile_ARCHITECTURE PROPERTY STRINGS all gfx803 gfx900 gfx906:xnack- gfx908:xnack-)
  set_property( CACHE Tensile_ARCHITECTURE PROPERTY STRINGS all gfx906:xnack- gfx908:xnack-)
  # -DTensile_ARCHITECTURE= set by install.sh for builds with and without tensile

  # Detect if target ID syntax if supported for default all AMDGPU_TARGETS list
  if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
      OUTPUT_VARIABLE CXX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
  endif()
  if(TARGET_ID_SUPPORT)
	  #Sanjay
	  #set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-")
	  set(target_list "gfx906:xnack-;gfx908:xnack-")
  else()
	  #Sanjay
	  #set(target_list "gfx803;gfx900;gfx906;gfx908")
	  set(target_list "gfx906;gfx908")
  endif()
  set(ALL_TARGETS ${target_list} CACHE STRING "List of specific machine types for library to target")

  # gpu architectures to build for
  if (Tensile_ARCHITECTURE)
    if( Tensile_ARCHITECTURE STREQUAL "all" )
      set( gpus ${ALL_TARGETS})
    else()
      set( gpus ${Tensile_ARCHITECTURE})
    endif()
    # match tensile if architecture is set
    set( AMD_GPUS ${gpus} )
    # must FORCE set this AMDGPU_TARGETS before any find_package( hip ...), via tensile or in this file
    # to override CACHE var and set cuda-gpu-arch flags via hip-config.cmake hip::device dependency
    set( AMDGPU_TARGETS "${AMD_GPUS}" CACHE STRING "AMD GPU targets to compile for" FORCE )
  endif()
  message(STATUS "DEBUG-4 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

endif()

# Find HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
	find_package( hip REQUIRED CONFIG PATHS ${USE_ROCM} )
endif( )

message(STATUS "Using AMDGPU_TARGETS: ${AMDGPU_TARGETS}")
foreach(amdgpu_target ${AMDGPU_TARGETS})
  set(CMAKE_CXX_FLAGS " --amdgpu-target=${amdgpu_target} ${CMAKE_CXX_FLAGS} ")
endforeach()
message(STATUS "DEBUG-5 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

####### USE_ROCM
####

include(caffe/cmake/Utils.cmake)
if (NOT DEFINED USE_ROCM)
   include(caffe/cmake/Cuda.cmake)
endif()

#our version of caffe needs to pull in headers from gnina and we'll use c++11
include_directories(.)

add_subdirectory(caffe)
add_dependencies(caffe libmolgrid)

include_directories(${CMAKE_BINARY_DIR}/caffe/include/)
include_directories(caffe/include)
add_subdirectory(gninasrc)

include_directories(${LIBMOLGRID_INCLUDE_DIR})


# enable testing
include(CTest)
enable_testing()
# define tests

add_subdirectory(test)
