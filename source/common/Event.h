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