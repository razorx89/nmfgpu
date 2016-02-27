#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	/** Wrapper class for CUDA Events to support automatic creation and destruction. Also the event API is wrapped
	into easier to use functions. */
	class Event {
		/** Raw CUDA event object. */
		cudaEvent_t m_event;

		/** Destroys the event object if one is still present. */
		void destroy();

		/** Releases the event object and sets the internal pointer to nullptr.
		@returns Valid event object which does not get managed by this instance anymore. */
		cudaEvent_t release();

	public:
		/** Measures the elapsed time between two recorded events.
		@param start Event which has recorded the starting point of the timespan.
		@param end Event which has recorded the end point of the timespan.
		@returns Elapsed time between both events. */
		static float getElapsedTime(const Event& start, const Event& end);

		/** Creates a new standard event object. */
		Event();

		/** Creates a new event object using the supplied flags. */
		Event(unsigned flags);

		/** Destroys the event object, if one is still present. */
		~Event();

		/** Move constructs from an existing event object.
		@param other Other event object instance. */
		Event(Event&& other);

		/** Move assigns from an other event object. An existing event object will be destroyed before
		assigning the new one.
		@param other Other event object instance. */
		Event& operator=(Event&& other);

		/** Implicitly converts the Event wrapper class to the underlying CUDA datatype. */
		operator cudaEvent_t() const {
			return m_event;
		}

		void record(cudaStream_t stream = nullptr);

		void synchronize();
	};
}