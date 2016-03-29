/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (svenkoitka@fh-dortmund.de)

This file is part of nmfgpu.

nmfgpu is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nmfgpu is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nmfgpu.  If not, see <http://www.gnu.org/licenses/>.
*/

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