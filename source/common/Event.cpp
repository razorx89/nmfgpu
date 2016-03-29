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