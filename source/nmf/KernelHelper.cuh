#pragma once

#if __CUDA_ARCH__ >= 350
#	define CUDA_READ_ONLY_CACHE(x) __ldg(&x)
#else
#	define CUDA_READ_ONLY_CACHE(x) x
#endif

namespace nmfgpu {
	template<typename Type, unsigned ThreadCount = 32>
	__device__ inline Type sumWarpReduction(Type var) {
		if (ThreadCount > 16)
			var += __shfl_xor(var, 16);
		if (ThreadCount > 8)
			var += __shfl_xor(var, 8);
		if (ThreadCount > 4)
			var += __shfl_xor(var, 4);
		if (ThreadCount > 2)
			var += __shfl_xor(var, 2);
		var += __shfl_xor(var, 1);
		return var;
	}
}
