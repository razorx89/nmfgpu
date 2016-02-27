#pragma once

#include <nmfgpu.h>

namespace nmfgpu {

	struct DispatcherConfig {
		unsigned numIterations;
		unsigned numRuns;

		NmfThresholdType thresholdType;
		double thresholdValue;

		UserInterruptCallback userInterruptCallback;

		template<typename NumericType>
		DispatcherConfig(NmfDescription<NumericType>& description)
			: numIterations(description.numIterations)
			, numRuns(description.numRuns)
			, thresholdType(description.thresholdType)
			, thresholdValue(description.thresholdValue)
			, userInterruptCallback(description.callbackUserInterrupt) {
		}
	};

	class Dispatcher {
	public:
		virtual ~Dispatcher() { };

		virtual bool dispatch() = 0;
	};
}