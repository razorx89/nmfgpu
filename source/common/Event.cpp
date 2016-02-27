#include <common/Event.h>

namespace nmfgpu {
	/* static */ float Event::getElapsedTime(const Event& start, const Event& end) {
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, end);
		return 0.0;
	}

	Event::Event() {
		// Create new event object
		cudaEventCreate(&m_event);
	}

	Event::Event(unsigned flags) {
		// Create new event object with flags
		cudaEventCreateWithFlags(&m_event, flags);
	}

	Event::~Event() {
		destroy();
	}

	Event::Event(Event&& other)
		: m_event(other.release()) { }

	Event& Event::operator=(Event&& other) {
		// Destroy the old event object and move the new one
		if (&other != this) {
			destroy();
			m_event = other.release();
		}

		return *this;
	}

	void Event::destroy() {
		// If an event object is available, then destroy it using the CUDA API
		if (m_event != nullptr) {
			cudaEventDestroy(m_event);
			m_event = nullptr;
		}
	}

	cudaEvent_t Event::release() {
		// Return the event object, but set the internal pointer to nullptr
		auto e = m_event;
		m_event = nullptr;
		return e;
	}

	void Event::record(cudaStream_t stream /* = nullptr */) {
		cudaEventRecord(m_event, stream);
	}

	void Event::synchronize() {
		cudaEventSynchronize(m_event);
	}
}