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