#include <common/Event.h>
#include <common/Stream.h>

namespace nmfgpu {
	void Stream::destroy() {
		if (m_stream != nullptr) {
			cudaStreamDestroy(m_stream);
			m_stream = nullptr;
		}
	}

	cudaStream_t Stream::release() {
		auto stream = m_stream;
		m_stream = nullptr;
		return stream;
	}

	Stream::Stream() {
		cudaStreamCreate(&m_stream);
	}

	Stream::Stream(unsigned flags) {
		cudaStreamCreateWithFlags(&m_stream, flags);
	}
	
	Stream::Stream(unsigned flags, unsigned priority) {
		cudaStreamCreateWithPriority(&m_stream, flags, priority);
	}

	Stream::~Stream() {
		destroy();
	}

	Stream::Stream(Stream&& other)
		: m_stream(other.release()) { }

	Stream& Stream::operator = (Stream&& other) {
		if (&other != this) {
			destroy();
			m_stream = other.release();
		}

		return *this;
	}

	void Stream::synchronize() {
		cudaStreamSynchronize(m_stream);
	}

	void Stream::waitFor(const Event& evt, unsigned flags /* = 0 */) {
		cudaStreamWaitEvent(m_stream, evt, flags);
	}
}