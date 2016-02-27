#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <nmf/FrobeniusResolver.h>

namespace nmfgpu {
	namespace details {
		template<typename NumericType>
		double resolveFrobenius(const HostMemory<NumericType>& hpsVTV, const HostMemory<NumericType>& hpsHTWTV, const HostMemory<NumericType>& hpsHHTWTW) {
			std::sort(&hpsHTWTV.get()[0], &hpsHTWTV.get()[hpsHTWTV.elements()]);
			std::sort(&hpsHHTWTW.get()[0], &hpsHHTWTW.get()[hpsHHTWTW.elements()]);

			auto resolvedFrob = 0.0;
			auto max = std::max(hpsVTV.elements(), std::max(hpsHTWTV.elements(), hpsHHTWTW.elements()));
			for (int j = 0; j < max; ++j) {
				if (j < hpsVTV.elements()) {
					resolvedFrob += hpsVTV.at(j);
				}

				if (j < hpsHTWTV.elements()) {
					resolvedFrob -= 2.f * hpsHTWTV.at(j);
				}

				if (j < hpsHHTWTW.elements()) {
					resolvedFrob += hpsHHTWTW.at(j);
				}
			}

			return std::sqrt(resolvedFrob);
		}
	}

	double resolveFrobenius(const HostMemory<float>& hpsVTV, const HostMemory<float>& hpsHTWTV, const HostMemory<float>& hpsHHTWTW) {
		return details::resolveFrobenius(hpsVTV, hpsHTWTV, hpsHHTWTW);
	}

	double resolveFrobenius(const HostMemory<double>& hpsVTV, const HostMemory<double>& hpsHTWTV, const HostMemory<double>& hpsHHTWTW) {
		return details::resolveFrobenius(hpsVTV, hpsHTWTV, hpsHHTWTW);
	}
}