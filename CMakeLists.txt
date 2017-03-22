cmake_minimum_required(VERSION 3.2.0)
project(nmfgpu CXX)

option(NMFGPU_COMPILE_UNIT_TESTS "Compile unit tests for nmfgpu" OFF)
set(NMFGPU_CUDA_ARCH "Kepler" CACHE STRING "CUDA architectures to compile for; one or more of Tesla, Fermi, Kepler" )
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  
#set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")

# Force an out-of-source build
string( COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" NMFGPU_COMPILE_INPLACE )
if( NMFGPU_COMPILE_INPLACE )
	message(FATAL_ERROR "Compiling NMFGPU with CMake requires an out-of-source build. To proceed:
  rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
  mkdir build
  cd build
  cmake ..
  make" )
endif()

			
#set ( CMAKE_SUPPRESS_REGENERATION true )
			
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
#message(${CMAKE_MODULE_PATH})



# Configure warnings for different compilers
if ( WIN32 )
	string( REGEX REPLACE " */W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 -MP" )
else()
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" )
endif()

# Configure C++11 support for non windows compilers
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif()






# Locate CUDA libraries
find_package( CUDA )
if ( CUDA_FOUND )
	message( "-- Found CUDA ${CUDA_VERSION}" )
	message( "   CUDA_INCLUDE_DIRS:   ${CUDA_INCLUDE_DIRS}"   )
	message( "   CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}" )
	include_directories( ${CUDA_INCLUDE_DIRS} )
	
	if(${CUDA_VERSION_MAJOR} LESS 7)
		message(FATAL_ERROR "nmfgpu requires at least CUDA 7.0 to compile!")
	endif()

	# NVCC options for the different cards
	# sm_xx is binary, compute_xx is PTX for forward compatability
	# MIN_ARCH is lowest requested version 
	# NV_SM    accumulates sm_xx for all requested versions
	# NV_COMP  is compute_xx for highest requested version
	set( NV_SM    "" )
	set( NV_COMP  "" )
	
	message( "   NMFGPU_CUDA_ARCH ${NMFGPU_CUDA_ARCH}" )
	if ( ${NMFGPU_CUDA_ARCH} MATCHES Tesla )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 130 )
		endif()
		# -gencode arch=compute_10,code=compute_10
		set( NV_SM "${NV_SM} -gencode arch=compute_13,code=sm_13" )
		set( NV_COMP        "-gencode arch=compute_13,code=compute_13" )
		message( "   compile for CUDA arch 1.3 (Tesla)" )
	endif()
	
	if ( ${NMFGPU_CUDA_ARCH} MATCHES Fermi )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 200 )
		endif()
		set( NV_SM "${NV_SM} -gencode arch=compute_20,code=sm_20" )
		set( NV_COMP        "-gencode arch=compute_20,code=compute_20" )
		message( "   compile for CUDA arch 2.x (Fermi)" )
	endif()
	
	if ( ${NMFGPU_CUDA_ARCH} MATCHES Kepler )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 300 )
		endif()
		set( NV_SM "${NV_SM} -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35" )
		set( NV_COMP        "-gencode arch=compute_35,code=compute_35" )
		message( "   compile for CUDA arch 3.x (Kepler)" )
	endif()
	
	if ( NOT MIN_ARCH )
		message( FATAL_ERROR "NMFGPU_CUDA_ARCH must contain one or more of Tesla, Fermi, Kepler." )
	endif()
	
	set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DHAVE_CUBLAS ${NV_SM} ${NV_COMP}" )
	add_definitions( "-DHAVE_CUBLAS -DMIN_CUDA_ARCH=${MIN_ARCH}" )
		
	if ( CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" )
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
		set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	endif()
	
	set(LIBS ${LIBS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})

	# cuSOLVER from CUDA >= 8.0 requires OpenMP
	if(NOT (${CUDA_VERSION_MAJOR} LESS 8))
		# OS X maybe also requires OpenMP, but cannot be tested right now  
		if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" )
			set(LIBS ${LIBS} gomp)
		endif()
	endif()
endif()

#####################################################################################################################
# Compile library, examples and unit tests 																			#
#####################################################################################################################
include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/source")

add_subdirectory(doc)
add_subdirectory(source)
add_subdirectory(example)

# If install prefix is different to the source directory, then the include directory must be copied as well
string(COMPARE NOTEQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_INSTALL_PREFIX}" NMFGPU_INSTALL_INCLUDE_FILES)
if(NMFGPU_INSTALL_INCLUDE_FILES)
	install(FILES ${PROJECT_SOURCE_DIR}/include/nmfgpu.h DESTINATION include)
endif()
