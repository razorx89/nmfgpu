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

#pragma once

#include <cuda_runtime_api.h>

namespace nmfgpu {
	// Forward-Declaration
	class Event;

	/** Wrapper class for CUDA Streams to support automatic creation and destruction. Also the stream API is wrapped
	into easier to use functions. */
	class Stream {
		/** Raw CUDA stream object. */
		cudaStream_t m_stream;

		/** Destroys the stream object if one is still present. */
		void destroy();
		
		/** Releases the stream object and sets the internal pointer to nullptr
		@returns Valid stream object which does not get managed by this instance anymore. */
		cudaStream_t release();

	public:
		/** Creates a new standard stream object. */
		Stream();

		/** Creates a new stream object using the supplied flags.
		@param flags Flags to specify the stream object type and capabilities. */
		Stream(unsigned flags);

		/** Creates a new stream object using the supplied flags and priority.
		@param flags Flags to specify the stream object type and capabilities.
		@param priority Priority of the stream. */
		Stream(unsigned flags, unsigned priority);
		
		/** Destroys the stream object, if one is still present. */
		~Stream();

		/** Move constructs from an existing stream object.
		@param other Other stream object instance. */
		Stream(Stream&& other);
		
		/** Move assigns from an other stream object. An existing stream object will be destroyed before
		assigning the new one. 
		@param other Other stream object instance. */
		Stream& operator = (Stream&& other);
		
		/** Implicitly converts the Stream wrapper class to the underlying CUDA datatype. */
		operator cudaStream_t() const {
			return m_stream;
		}

		/** Synchronizes the CPU execution with the CUDA stream. Further CPU execution will halt
		until the stream has processed all pending operations. */
		void synchronize();
		
		/** Stream will wait for the specified event to be recorded, after all pending operations are processed. 
		@param evt Event which has to be recorded before further execution of the stream will continue. 
		@param flags Additional flags to specify the wait operation. */
		void waitFor(const Event& evt, unsigned flags = 0);
	};
}