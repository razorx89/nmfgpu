SET(HEADERS_INCLUDE ${PROJECT_SOURCE_DIR}/include/nmfgpu.h)
SET(HEADERS_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/common/Event.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Interface.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Logging.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Matrix.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/MatrixDescription.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Memory.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Misc.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Stream.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Traits.h
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Wrapper.h)
SET(HEADERS_INIT ${CMAKE_CURRENT_SOURCE_DIR}/init/CopyStrategy.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/EInNMF.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/InitializationStrategy.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/KernelMeanColumn.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/KMeansStrategy.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/MeanColumnStrategy.h
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/RandomValueStrategy.h)
SET(HEADERS_KMEANS ${CMAKE_CURRENT_SOURCE_DIR}/kmeans/kMeans.h)
SET(HEADERS_NMF ${CMAKE_CURRENT_SOURCE_DIR}/nmf/Algorithm.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/Context.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/AlgorithmAlternatingHoyerConstrainedLeastSquares.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/AlgorithmAlternatingLeastSquares.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/AlgorithmGradientDescentConstrainedLeastSquares.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/AlgorithmMultiplicativeFrobenius.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/AlgorithmNonSmoothNMF.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/Dispatcher.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/FrobeniusResolver.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelFillMatrix.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelMakeNonNegative.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelMultiplyDivide.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelNormalizeColumns.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelTraceMultiplication.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelHelper.cuh
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/SingleGpuDispatcher.h
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/Summary.h)

SET(HEADERS ${HEADERS_INCLUDE} ${HEADERS_COMMON} ${HEADERS_INIT} ${HEADERS_KMEANS} ${HEADERS_NMF})

SET(SOURCES_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/common/Event.cpp
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Interface.cpp
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Logging.cpp
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Matrix.cpp
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Stream.cpp
				   ${CMAKE_CURRENT_SOURCE_DIR}/common/Wrapper.cpp)
SET(SOURCES_INIT ${CMAKE_CURRENT_SOURCE_DIR}/init/CopyStrategy.cpp
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/EInNMF.cu
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/InitializationStrategy.cpp
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/KernelMeanColumn.cu
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/KMeansStrategy.cpp
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/MeanColumnStrategy.cpp
				 ${CMAKE_CURRENT_SOURCE_DIR}/init/RandomValueStrategy.cpp)
SET(SOURCES_KMEANS ${CMAKE_CURRENT_SOURCE_DIR}/kmeans/kMeans.cu)
SET(SOURCES_NMF	${CMAKE_CURRENT_SOURCE_DIR}/nmf/Algorithm.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/FrobeniusResolver.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelFillMatrix.cu
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelMakeNonNegative.cu
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelMultiplyDivide.cu
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelNormalizeColumns.cu
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/KernelTraceMultiplication.cu
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/SingleGpuDispatcher.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/nmf/Summary.cpp)
SET(SOURCES_KERNELS )

SET(SOURCES ${SOURCES_COMMON} ${SOURCES_INIT} ${SOURCES_KMEANS} ${SOURCES_NMF})
			
					
SOURCE_GROUP("Header Files" FILES ${HEADERS_INCLUDE})
SOURCE_GROUP("Header Files\\common" FILES ${HEADERS_COMMON})
SOURCE_GROUP("Source Files\\common" FILES ${SOURCES_COMMON})
SOURCE_GROUP("Header Files\\init" FILES ${HEADERS_INIT})
SOURCE_GROUP("Source Files\\init" FILES ${SOURCES_INIT})
SOURCE_GROUP("Header Files\\kmeans" FILES ${HEADERS_KMEANS})
SOURCE_GROUP("Source Files\\kmeans" FILES ${SOURCES_KMEANS})
SOURCE_GROUP("Header Files\\nmf" FILES ${HEADERS_NMF})
SOURCE_GROUP("Source Files\\nmf" FILES ${SOURCES_NMF})

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BIT_POSTFIX 64)
else()
	set(BIT_POSTFIX 32)
endif()

cuda_add_library(nmfgpu_static STATIC ${SOURCES} ${HEADERS})
target_link_libraries(nmfgpu_static ${LIBS})
set_target_properties(nmfgpu_static PROPERTIES COMPILE_DEFINITIONS "NMFGPU_STATIC_LINKING")
set_target_properties(nmfgpu_static PROPERTIES OUTPUT_NAME "nmfgpu${BIT_POSTFIX}s")

cuda_add_library(nmfgpu SHARED ${SOURCES} ${HEADERS})
target_link_libraries(nmfgpu ${LIBS})
set_target_properties(nmfgpu PROPERTIES COMPILE_DEFINITIONS "NMFGPU_EXPORTING")
set_target_properties(nmfgpu PROPERTIES OUTPUT_NAME "nmfgpu${BIT_POSTFIX}")

install(TARGETS nmfgpu DESTINATION lib)
install(TARGETS nmfgpu RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
install(TARGETS nmfgpu_static DESTINATION lib)
#install(TARGETS nmfgpu RUNTIME DESTINATION "D:/Studium/nmfgpu4R/inst/libs/x64")
