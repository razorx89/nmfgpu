#pragma once

#include <common/Logging.h>
#include <common/Traits.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

namespace nmfgpu {
	namespace traits {
		/** Checks for a static allocate and deallocate interface. 
		@tparam DataType Underlying data type of the allocator interface. */
		template<typename DataType>
		struct CheckForStaticAllocaterInterface {
			/** SFINAE construction to check for "static DataType* allocate(size_t)" and "static void deallocate(DataType*)". */
			template<typename T,
				DataType*(*)(size_t) = &T::allocate,
				void(*)(DataType*) = &T::deallocate>
			struct get { };
		};

		/** Allocates page-locked memory on the host.
		@tparam DataType Underlying data type of the allocated memory. */
		template<typename DataType>
		class HostMemoryAllocator {
			/** No constructor available due to static interface. */
			HostMemoryAllocator() = delete;

		public:
			/** Allocates page-locked memory of sufficient size to hold the requested element count of DataType.
			@param elements Element count which should be allocated.
			@returns Page-locked memory which is able to hold the requested amount of elements. */
			static DataType* allocate(size_t elements) {
				DataType* memory = nullptr;
				CUDA_CALL(cudaMallocHost(reinterpret_cast<void**>(&memory), elements * sizeof(DataType)));
				return memory;
			}

			/** Deallocates page-locked memory which was allocated using the allocate interface.
			@param memory Pointer to the page-locked memory which should be deallocated. */
			static void deallocate(DataType* memory) {
				CUDA_CALL(cudaFreeHost(memory));
			}
		};

		/** Allocates memory on the CUDA device. 
		@tparam DataType Underlying data type of the allocated memory. */
		template<typename DataType>
		class DeviceMemoryAllocator {
			/** No constructor available due to static interface. */
			DeviceMemoryAllocator() = delete;

		public:
			/** Allocates device memory of sufficient size to hold the requested element count of DataType. 
			@param elements Element count which should be allocated.
			@returns Device memory which is able to hold the requested amount of elements. */
			static DataType* allocate(size_t elements) {
				DataType* memory = nullptr;
				CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&memory), elements * sizeof(DataType)));
				return memory;
			}

			/** Deallocates device memory which was allocated using the allocate interface.
			@param memory Pointer to the device memory which should be deallocated. */
			static void deallocate(DataType* memory) {
				CUDA_CALL(cudaFree(memory));
			}
		};
	}

	class IMemory {
	public:
		virtual ~IMemory() { };
		virtual void* raw() const = 0;
		virtual size_t elements() const = 0;
	};

	template<typename DataType, typename Allocator, typename EnableIf = void>
	class Memory : public IMemory { };
	
	/** Manages memory using a custom allocator of any requested data type.
	@tparam DataType Underlying data type which should be used to determine the required amount of memory.
	@tparam Allocator Class or struct which has two static functions. The first "DataType* allocate(size_t)" is used to allocate the 
	required memory. The second "void deallocate(DataType*)" is used to deallocate the allocated memory. Existence of both interface
	functions will be ensured using type traits. */
	template<typename DataType, typename Allocator>
	class Memory<DataType, Allocator, typename std::enable_if<traits::HasMember<Allocator, traits::CheckForStaticAllocaterInterface<DataType>>::value>::type> : public IMemory {
		/** Information about allocated memory and size, used to be allocated as shared memory between different instances. */
		struct Storage {
			/** Contains the element count which is available in the allocated memory. */
			size_t elements{ 0 };

			/** Pointer to the allocated memory. */
			DataType* memory{ nullptr };

			/** Deallocates any remaining memory. */
			~Storage() {
				Allocator::deallocate(memory);
			}
		};
		
		/** Storage object containing the allocated memory and information of element count. */
		std::shared_ptr<Storage> m_storage;
		
	public:
		Memory()
			: m_storage(nullptr) { }

		/** Initializes the memory class by allocating enough memory to store the elements.
		@param elements Element count to be allocated.
		@throws std::invalid_argument Will be thrown if elements is equal to zero. */
		Memory(size_t elements)
			: m_storage(std::make_shared<Storage>()) {
			// If zero elements was specified, then it makes no sense to allocate memory
			if (elements == 0u) {
				throw std::invalid_argument("Zero elements specified");
			}

			// Allocate a chunk of memory for specified element count
			m_storage->memory = Allocator::allocate(elements);
			m_storage->elements = elements;
		}

		/** Move constructs from an existing memory instance.
		@param other Other memory object instance. */
		Memory(Memory&& other)
			: m_storage(std::move(other.m_storage)) {
		}

		bool allocate(size_t elements) {
			// If zero elements was specified, then it makes no sense to allocate memory
			if (elements == 0u) {
				throw std::invalid_argument("Zero elements specified");
			}

			// Allocate a chunk of memory for specified element count
			m_storage = std::make_shared<Storage>();
			m_storage->memory = Allocator::allocate(elements);
			m_storage->elements = elements;

			return m_storage->memory != nullptr;
		}

		/** Move assigns from an other memory object instance. 
		@param other Other memory object instance. */
		Memory& operator = (Memory&& other) {
			if (&other != this) {
				m_storage = std::move(other.m_storage);
			}

			return *this;
		}

		virtual ~Memory() { };

		/** Gets the total amount of bytes allocated. 
		@returns Amount of bytes allocated. */
		size_t bytes() const {
			return m_storage->elements * sizeof(DataType);
		}

		/** Gets the total amount of elements, which can be stored in the allocated memory.
		@returns Amount of elements.*/
		virtual size_t elements() const override {
			return m_storage->elements;
		}
		
		/** Gets the pointer to the allocated memory.
		@returns Pointer to the allocated memory. */
		DataType* get() const {
			return m_storage->memory;
		}

		/** Gets a reference to the requested element within the allocated memory.
		@param offset Offset to the requested element.
		@returns Reference to the requested element. */
		DataType& at(size_t offset) const {
			// Do some range checking
			if (offset >= m_storage->elements) {
				throw std::out_of_range("offset >= elements");
			}

			// Return reference to requested element
			return m_storage->memory[offset];
		}
		
		/** Returns the current reference count of the allocated storage.
		@returns Reference count of the allocated storage. */
		size_t references() const {
			return size_t(m_storage.use_count());
		}

		virtual void* raw() const override {
			return reinterpret_cast<void*>(m_storage->memory);
		}
	};

	namespace details {
		template<typename DataType, typename AllocatorDst, typename AllocatorSrc>
		void copyMemoryToMemory(Memory<DataType, AllocatorDst>& dst, const Memory<DataType, AllocatorSrc>& src, cudaMemcpyKind kind, cudaStream_t stream) {
			CUDA_CALL(cudaMemcpyAsync(dst.get(), src.get(), dst.bytes(), kind, stream));
		}
	}

	template<typename DataType>
	class DeviceMemory;

	template<typename DataType>
	class HostMemory : public Memory<DataType, traits::HostMemoryAllocator<DataType>> {
	public:
		HostMemory() = default;
		HostMemory(size_t elements)
			: Memory<DataType, traits::HostMemoryAllocator<DataType>>(elements) { }

		void copyTo(HostMemory<DataType>& dst, cudaStream_t stream = nullptr) {
			details::copyMemoryToMemory(dst, *this, cudaMemcpyHostToHost, stream);
		}

		void copyTo(DeviceMemory<DataType>& dst, cudaStream_t stream = nullptr) {
			details::copyMemoryToMemory(dst, *this, cudaMemcpyHostToDevice, stream);
		}
	};

	template<typename DataType>
	class DeviceMemory : public Memory<DataType, traits::DeviceMemoryAllocator<DataType>> {
	public:
		DeviceMemory() = default;
		DeviceMemory(size_t elements)
			: Memory<DataType, traits::DeviceMemoryAllocator<DataType>>(elements) { }

		//virtual ~DeviceMemory() = default;

		void copyTo(HostMemory<DataType>& dst, cudaStream_t stream = nullptr) {
			details::copyMemoryToMemory(dst, *this, cudaMemcpyDeviceToHost, stream);
		}

		void copyTo(DeviceMemory<DataType>& dst, cudaStream_t stream = nullptr) {
			details::copyMemoryToMemory(dst, *this, cudaMemcpyDeviceToDevice, stream);
		}
	};
}