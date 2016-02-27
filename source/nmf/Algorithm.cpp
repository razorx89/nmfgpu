#include <nmf/Algorithm.h>
#include <random>

namespace nmfgpu {
	IAlgorithm::IAlgorithm(unsigned seed) 
		: m_randomGenerator(std::bind(std::uniform_int_distribution<unsigned>(0, std::numeric_limits<unsigned>::max()), std::mt19937(seed))) { }

	unsigned IAlgorithm::generateRandomNumber() {
		return m_randomGenerator();
	}
}