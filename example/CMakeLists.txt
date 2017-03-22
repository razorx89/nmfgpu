cuda_add_executable(nmfgpu_example main.cpp)
target_link_libraries(nmfgpu_example nmfgpu_static)
install(TARGETS nmfgpu_example DESTINATION bin)
