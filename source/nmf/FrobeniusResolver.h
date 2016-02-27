#pragma once

#include <common/Memory.h>

namespace nmfgpu {
	double resolveFrobenius(const HostMemory<float>& hpsVTV, const HostMemory<float>& hpsHTWTV, const HostMemory<float>& hpsHHTWTW);
	double resolveFrobenius(const HostMemory<double>& hpsVTV, const HostMemory<double>& hpsHTWTV, const HostMemory<double>& hpsHHTWTW);
}