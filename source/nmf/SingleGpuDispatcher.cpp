#include <algorithm>
#include <assert.h>
#include <cmath>
#include <common/Interface.h>
#include <common/Logging.h>
#include <cuda_runtime_api.h>
#include <init/InitializationStrategy.h>
#include <iostream>
#include <nmf/Algorithm.h>
#include <nmf/SingleGpuDispatcher.h>
#include <nmf/Summary.h>

namespace nmfgpu {
	SingleGpuDispatcher::SingleGpuDispatcher(const DispatcherConfig& dispatcherConfig, std::unique_ptr<IAlgorithm>& algorithm, Summary* statistic)
		: m_config(dispatcherConfig)
		, m_algorithm(std::move(algorithm)) 
		, m_summary(statistic) {
		assert(m_algorithm != nullptr);
	}

	SingleGpuDispatcher::~SingleGpuDispatcher() = default;

	void SingleGpuDispatcher::printHeader(bool multipleRuns) {
		if (!Logging::instance().summary().allowed()) {
			return;
		}

		if (multipleRuns) {
			Logging::instance().summary()
				.print(" -------------------------------------------------------------------------------------------------------------").lineFeed()
				.print(" |   Run   | Iteration |     Frobenius     |       RMSD       |       Delta      | Elapsed Time |   Status   |").lineFeed()
				.print(" -------------------------------------------------------------------------------------------------------------").lineFeed();
		} else {
			Logging::instance().summary()
				.print(" ---------------------------------------------------------------------------------------------------").lineFeed()
				.print(" | Iteration |     Frobenius     |       RMSD       |       Delta      | Elapsed Time |   Status   |").lineFeed()
				.print(" ---------------------------------------------------------------------------------------------------").lineFeed();
		}
	}

	namespace details {
		template<int N>
		void generateDurationString(char(&buffer)[N], int64_t milliseconds) {
			auto h = milliseconds / 3600000;
			milliseconds %= 3600000;
			auto m = milliseconds / 60000;
			milliseconds %= 60000;
			auto s = milliseconds / 1000;
			auto ms = milliseconds % 1000;
			sprintf(buffer, "%02d:%02d:%02d.%03d", int(h), int(m), int(s), int(ms));
		}

		void generateMarqueeProgress(char* buffer, unsigned charCount, unsigned i) {
			std::fill(&buffer[0], &buffer[charCount], ' ');
			buffer[i % charCount] = '<';
			buffer[(i + 1) % charCount] = '=';
			buffer[(i + 2) % charCount] = '>';
		}
	}

	void SingleGpuDispatcher::printIterationInfo(bool multipleRuns, unsigned run, unsigned iteration, double frobenius, double rmsd, double delta, std::chrono::milliseconds elapsedTime) {
		if (!Logging::instance().summary().allowed()) {
			return;
		}
	
		// Generate string of elapsed time
		char bufferTime[32];
		details::generateDurationString(bufferTime, elapsedTime.count());

		// Generate marquee string
		char bufferMarquee[16]{'\0'};
		details::generateMarqueeProgress(bufferMarquee, 10, iteration / ITERATION_BATCH_COUNT);

		// Print information about current iteration
		char buffer[1024];
		bool clearLine = iteration != ITERATION_BATCH_COUNT;
		if (multipleRuns) {
			sprintf(buffer, "%s | %7d | %9d | %17.4f | %16.4f | %16.4f | %s | %s |", clearLine ? "\r" : "", run, iteration, frobenius, rmsd, delta, bufferTime, bufferMarquee);
		} else {
			sprintf(buffer, "%s | %9d | %17.4f | %16.4f | %16.4f | %s | %s |", clearLine ? "\r" : "", iteration, frobenius, rmsd, delta, bufferTime, bufferMarquee);
		}

		Logging::instance().summary().print(buffer);
	}

	void SingleGpuDispatcher::printIterationFinalInfo(bool multipleRuns, unsigned run, unsigned iteration, double frobenius, double rmsd, double delta, std::chrono::milliseconds elapsedTime, const char* status) {
		if (!Logging::instance().summary().allowed()) {
			return;
		}

		// Generate string of elapsed time
		char bufferTime[32];
		details::generateDurationString(bufferTime, elapsedTime.count());

		// Print final information about the run
		char buffer[1024];
		if (multipleRuns) {
			sprintf(buffer, "\r | %7d | %9d | %17.4f | %16.4f | %16.4f | %s | %10s |\n", run, iteration, frobenius, rmsd, delta, bufferTime, status);
			Logging::instance().summary().print(buffer);

			if (run == m_config.numRuns) {
				Logging::instance().summary().print(" -------------------------------------------------------------------------------------------------------------").lineFeed();
			}
		} else {
			sprintf(buffer, "\r | %9d | %17.4f | %16.4f | %16.4f | %s | %10s |\n", iteration, frobenius, rmsd, delta, bufferTime, status);
			Logging::instance().summary().print(buffer, " ---------------------------------------------------------------------------------------------------").lineFeed();
		}

	}

	bool SingleGpuDispatcher::dispatch() {
		// Reset summary if available
		if (m_summary != nullptr) {
			m_summary->reset();
		}

		// Initialize the matrices
		m_algorithm->allocateMemory();
		//m_initializationStrategy->initialize();

		// Define a helper for checking user interrupt
		auto checkUserInterrupt = [&]() -> bool {
			return m_config.userInterruptCallback != nullptr && m_config.userInterruptCallback();
		};

		// Print algorithm header
		Logging::instance().summary()
			.print(" Executing ", m_config.numRuns, " run(s) of the '", m_algorithm->name(), "' algorithm on CUDA device #", g_context->deviceID, ": ")
			.lineFeed();

		// Perform specified amount of runs
		auto userInterruptOccured = false;
		double bestErrorRun = std::numeric_limits<double>::max();
		for (auto run = 1u; run <= m_config.numRuns; ++run) {
			// Print iteration status header			
			if (run == 1u) {
				printHeader(m_config.numRuns > 1u);
			}

			// Initialize factorization matrices
			m_algorithm->initialize();
			cudaDeviceSynchronize();

			// Run loop until maximum iterations are reached, the user interrupts or a break through convergence occures
			auto algorithmStarted = std::chrono::high_resolution_clock::now();
			auto elapsedTime = std::chrono::milliseconds();
			auto lastError = 0.0;
			auto iteration = 1u;
			auto delta = 0.0;
			for (; iteration <= m_config.numIterations && !(userInterruptOccured = checkUserInterrupt()); ++iteration) {
				// Determine if remaining error has to be computed
				bool computeError = iteration % ITERATION_BATCH_COUNT == 0 || iteration == m_config.numIterations;

				// Do one iteration
				m_algorithm->computeIteration(computeError);

				// Check for convergence
				if (computeError) {
					// Measure elapsed time
					elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - algorithmStarted);

					// Get current remaining error
					auto currentError = 0.0;
					if (m_config.thresholdType == NmfThresholdType::Frobenius) {
						currentError = m_algorithm->frobeniusNorm();
					} else {
						currentError = m_algorithm->rmsd();
					}

					// Compute delta to last convergence check
					delta = currentError - lastError;

					// Print current status
					printIterationInfo(m_config.numRuns > 1, run, iteration, m_algorithm->frobeniusNorm(), m_algorithm->rmsd(), delta, elapsedTime);

					if (lastError != 0.0 && fabs(delta) < m_config.thresholdValue) {
						break;
					}
					lastError = currentError;
				}
			}

			// Ensure that last increment of for loop does not account
			iteration = std::min(iteration, m_config.numIterations);

			if (userInterruptOccured) {
				printIterationFinalInfo(m_config.numRuns > 1, run, iteration, m_algorithm->frobeniusNorm(), m_algorithm->rmsd(), delta, elapsedTime, "Aborted");
				break;
			}

			// Check if achieved remaining error of the factorization is better than the from a previous run
			auto stored = false;
			if (m_algorithm->frobeniusNorm() < bestErrorRun) {
				// Save all information about this run
				if (m_summary != nullptr) {
					auto record = ExecutionRecord();
					record.elapsedTime = elapsedTime.count() / 1000.0;
					record.frobenius = m_algorithm->frobeniusNorm();
					record.rmsd = m_algorithm->rmsd();
					record.numIterations = iteration;
					m_summary->insert(record);
				}

				m_algorithm->storeFactorization();
				bestErrorRun = m_algorithm->frobeniusNorm();
				stored = true;

			} else {
				stored = false;
			}

			// Print final status
			printIterationFinalInfo(m_config.numRuns > 1, run, iteration, m_algorithm->frobeniusNorm(), m_algorithm->rmsd(), delta, elapsedTime, stored ? "Stored" : "Discarded");
		}

		cudaDeviceSynchronize();
		m_algorithm->deallocateMemory();

		return !userInterruptOccured;
	}
}