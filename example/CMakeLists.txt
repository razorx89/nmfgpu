cuda_add_executable(nmfgpu_example main.cpp)
target_link_libraries(nmfgpu_example nmfgpu_static)
# cuSOLVER from CUDA >= 8.0 requires OpenMP
if(NOT (${CUDA_VERSION_MAJOR} LESS 8))
	# OS X maybe also requires OpenMP, but cannot be tested right now  
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" )
		target_link_libraries(nmfgpu_example gomp)
	endif()
endif()
install(TARGETS nmfgpu_example DESTINATION bin)
