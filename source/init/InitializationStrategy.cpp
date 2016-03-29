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

#include <common/Matrix.h>
#include <common/Misc.h>
#include <init/CopyStrategy.h>
#include <init/InitializationStrategy.h>
#include <init/KMeansStrategy.h>
#include <init/MeanColumnStrategy.h>
#include <init/RandomValueStrategy.h>
#include <nmf/Algorithm.h>
#include <nmf/Context.h>
#include <nmfgpu.h>

namespace nmfgpu { 
	namespace details {
		template<typename NumericType>
		std::unique_ptr<InitializationStrategy<NumericType>> create(const NmfDescription<NumericType>& description, const NMFCoreMatrices<NumericType>& coreMatrices) {
			switch (description.initMethod) {
			case NmfInitializationMethod::CopyExisting:				return make_unique<CopyStrategy<NumericType>>(description.outputMatrixW, description.outputMatrixH);
			case NmfInitializationMethod::AllRandomValues:			return make_unique<RandomValueStrategy<NumericType>>(description.seed);
			case NmfInitializationMethod::MeanColumns:				return make_unique<MeanColumnStrategy<NumericType>>(description.seed, coreMatrices.deviceV);
			case NmfInitializationMethod::KMeansAndRandomValues:
			case NmfInitializationMethod::KMeansAndAbsoluteWTV:
			case NmfInitializationMethod::KMeansAndNonNegativeWTV: 
			case NmfInitializationMethod::EInNMF:					return make_unique<KMeansStrategy<NumericType>>(description.seed, description.initMethod, coreMatrices.deviceV);
			default: return nullptr;
			}
		}
	}

	template<>
	/* static */ std::unique_ptr<InitializationStrategy<float>> InitializationStrategy<float>::create(const NmfDescription<float>& description, const NMFCoreMatrices<float>& coreMatrices) {
		return details::create(description, coreMatrices);
	}

	template<>
	/* static */ std::unique_ptr<InitializationStrategy<double>> InitializationStrategy<double>::create(const NmfDescription<double>& description, const NMFCoreMatrices<double>& coreMatrices) {
		return details::create(description, coreMatrices);
	}
}