/*
nmfgpu - CUDA accelerated computation of Non-negative Matrix Factorizations (NMF)

Copyright (C) 2015-2016  Sven Koitka (sven.koitka@fh-dortmund.de)

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

#pragma once

#include <chrono>
#include <memory>
#include <nmf/Dispatcher.h>
namespace nmfgpu {
	// Forward-Declarations
	class IAlgorithm;
	class IInitializationStrategy;
	class Summary;

	/** Dispatches an algorithm execution on a single GPU. The result of the algorithm will be written
	into the user supplied memory and the statistic will of the execution will be generated. Multiple runs in a monte-carlo like fashion can be
	performed and the best result will be returned. */
	class SingleGpuDispatcher : public Dispatcher {
		static const unsigned ITERATION_BATCH_COUNT = 10u;
		const DispatcherConfig& m_config;

		/** Instance of the algorithm to be executed once. */
		std::unique_ptr<IAlgorithm> m_algorithm;

		/** Statistic about the algorithm execution. */
		Summary* m_summary;

		void printHeader(bool multipleRuns);
		void printIterationInfo(bool multipleRuns, unsigned run, unsigned iteration, double frobenius, double rmsd, double delta, std::chrono::milliseconds elapsedTime);
		void printIterationFinalInfo(bool multipleRuns, unsigned run, unsigned iteration, double frobenius, double rmsd, double delta, std::chrono::milliseconds elapsedTime, const char* status);

	public:
		/** Initializes the dispatcher with an algorithm and the disptacher configuration.
		@param config
		@param algorithm Algorithm which gets executed. */
		SingleGpuDispatcher(const DispatcherConfig& config, std::unique_ptr<IAlgorithm>& algorithm, Summary* statistic);

		/** Empty destructor just to hide the destructor calls of forward declared classes. */
		virtual ~SingleGpuDispatcher();

		/** @copydoc IAlgorithmDispatcher::dispatch() */
		virtual bool dispatch() override;
	};
}