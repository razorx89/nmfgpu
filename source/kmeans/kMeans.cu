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

#include <algorithm>
#include <array>
#include <common/Interface.h>
#include <common/Logging.h>
#include <common/MatrixDescription.h>
#include <common/Memory.h>
#include <common/Stream.h>
#include <common/Wrapper.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <kmeans/kMeans.h>
#include <nmf/KernelHelper.cuh>
#include <numeric>
#include <random>
#include <set>

namespace nmfgpu {
	namespace details {
		template<typename NumericType>
		__device__ inline NumericType distanceSq(unsigned dataID, unsigned clusterID, unsigned numRows, NumericType* data, unsigned ldData, NumericType* clusters, unsigned ldClusters) {
			NumericType sum = 0.0;

			for (auto i = threadIdx.x; i < numRows; i += 32) {
				auto diff = data[dataID * ldData + i] - clusters[clusterID * ldClusters + i];
				sum += diff * diff;
			}

			return sumWarpReduction(sum);
		}

		/** Updates the membership of data entries to cluster centers. The algorithm assumes that blockDim.x == 32, so that one warp processes one column of data. */
		template<typename NumericType> 
		__global__ void kernelUpdateMembership(unsigned numData, unsigned numAttributes, unsigned numClusters, NumericType* data, unsigned ldData, NumericType* clusters, unsigned ldClusters, unsigned* dataClusterMembership, unsigned* membershipChangeCount) {
			// Get ID of corresponding data entry for this warp
			auto dataID = blockDim.y * blockIdx.x + threadIdx.y; 
			if (dataID >= numData) {
				return;
			}

			// Find cluster with the minimum distance
			auto bestCluster = 0u;
			auto bestClusterDistance = distanceSq(dataID, 0, numAttributes, data, ldData, clusters, ldClusters);
			for (auto clusterID = 1u; clusterID < numClusters; ++clusterID) {
				auto distance = distanceSq(dataID, clusterID, numAttributes, data, ldData, clusters, ldClusters);
				
				if (distance < bestClusterDistance) {
					bestClusterDistance = distance;
					bestCluster = clusterID;
				}
			}

			// Check if membership has changed
			if (dataClusterMembership[dataID] != bestCluster && threadIdx.x == 0) {
				dataClusterMembership[dataID] = bestCluster;
				atomicAdd(membershipChangeCount, 1);
			}
		}


		template<typename NumericType>
		__global__ void kernelUpdateClusterCenters(unsigned numAttributes, unsigned numClusters, NumericType* data, unsigned ldData, NumericType* clusters, unsigned ldClusters, unsigned* sortedDataClusterMembership, unsigned* sortedDataClusterMembershipEntryPoints, unsigned* sortedDataClusterMembershipEntryCount) {
			// Determine ID of processing cluster
			auto clusterID = blockDim.y * blockIdx.y + threadIdx.y;
			if (clusterID >= numClusters) {
				return;
			}

			// Does this cluster have any corresponding data entries?
			auto entryPoint = sortedDataClusterMembershipEntryPoints[clusterID];
			auto entryCount = sortedDataClusterMembershipEntryCount[clusterID];
			if (entryCount == 0) {
				return;
			}

			auto row = blockDim.x * blockIdx.x  * 2 + threadIdx.x;  
			if (row >= numAttributes) {
				return;
			}

			// Optimize array accesses
			clusters = &clusters[clusterID * ldClusters + row];
			data = &data[row];

			auto sum = NumericType();
			auto sum2 = NumericType();
			for (auto i = 0u; i < entryCount; ++i) {
				auto dataID = sortedDataClusterMembership[entryPoint + i];
				sum += data[dataID * ldData];
				if (row + 32 < numAttributes) {
					sum2 += data[dataID * ldData + 32];
				}
			}
			sum /= NumericType(entryCount);
			sum2 /= NumericType(entryCount);

			clusters[0] = sum;

			if (row + 32 < numAttributes) {
				clusters[32] = sum2;
			}
		}


		template<typename NumericType>
		void computeKMeans(const MatrixDescription<NumericType>& data, const MatrixDescription<NumericType>& clusters, DeviceMemory<unsigned>& dataClusterMembership, unsigned seed, unsigned maxiter, double threshold) {
			if (data.format != StorageFormat::Dense || clusters.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Input matrices must be stored in dense format!").lineFeed();
				return;
			}
			
			// Phase 1: Initialize cluster centers (Forgy) ------------------------------------------------------------------------------

			// Generate random indices for picking data columns
			auto randomIndices = std::vector<unsigned>(data.columns);
			auto generator = std::mt19937(seed);
			std::iota(randomIndices.begin(), randomIndices.end(), 0);
			std::shuffle(randomIndices.begin(), randomIndices.end(), generator);
			randomIndices.erase(randomIndices.begin() + clusters.columns, randomIndices.end());

			// Initialize the clusters in device memory using the random indices
			CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, nullptr));
			for (auto i = 0u; i < clusters.columns; ++i) {
				CUBLAS_CALL(wrapper::cublasXcopy(static_cast<int>(clusters.rows), &data.dense.values[randomIndices[i] * data.dense.leadingDimension], 1, &clusters.dense.values[i * clusters.dense.leadingDimension], 1));
			}

			// Phase 2: Perform cluster updates until convergence or maximum iteration count ------------------------------------
			auto deviceMembershipChangeCount = DeviceMemory<unsigned>(1);
			auto streams = std::array<Stream, 8>();
			auto hostDataClusterMembership = HostMemory<unsigned>(data.columns);

			auto hostSortedDataClusterMembership = HostMemory<unsigned>(data.columns);
			auto hostEntryPoints = HostMemory<unsigned>(clusters.columns);
			auto deviceEntryPoints = DeviceMemory<unsigned>(clusters.columns);
			auto hostEntryCount = HostMemory<unsigned>(clusters.columns);
			auto deviceEntryCount = DeviceMemory<unsigned>(clusters.columns);

			auto membershipInfo = std::vector<std::pair<unsigned, unsigned>>();
			membershipInfo.reserve(data.columns);

			auto iteration = 0u;
			auto percentageChange = 0.0;
			do {
				// Update membership of data entries
				auto membershipChangeCount = 0u;
				CUDA_CALL(cudaMemcpy(deviceMembershipChangeCount.get(), &membershipChangeCount, sizeof(unsigned), cudaMemcpyHostToDevice));
				
				auto blockDim = dim3(32, 4);
				auto gridDim = dim3((data.columns + blockDim.y - 1) / blockDim.y);
				details::kernelUpdateMembership<NumericType><<<gridDim, blockDim>>>(data.columns, data.rows, clusters.columns, data.dense.values, data.dense.leadingDimension, clusters.dense.values, clusters.dense.leadingDimension, dataClusterMembership.get(), deviceMembershipChangeCount.get());
				CUDA_CALL(cudaGetLastError());

				// Copy count of changed cluster centers
				CUDA_CALL(cudaMemcpy(&membershipChangeCount, deviceMembershipChangeCount.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
				percentageChange = membershipChangeCount / double(data.columns);

				if (Logging::instance().verbosity() == Verbosity::Debugging) {
					std::cout << " [DEBUG] k-Means: " << membershipChangeCount << "(" << percentageChange * 100 << "%) memberships have changed in iteration " << iteration << std::endl;
				}

				// Did any cluster change during update?
				if (membershipChangeCount > 0u) {
					// Copy membership information to host
					dataClusterMembership.copyTo(hostDataClusterMembership);
					cudaDeviceSynchronize();
#define USE_OPTIMIZED 1
					// Evaluate memberships to buckets of <clusterID[1], entryID[1..n]>
#if USE_OPTIMIZED == 0
					auto membershipBuckets = std::multimap<unsigned, unsigned>();
					for (auto i = 0u; i < data.columns(); ++i) {
						membershipBuckets.insert(std::make_pair(hostDataClusterMembership.at(i), i));
					}
#else
					membershipInfo.clear();
					for (auto i = 0u; i < data.columns; ++i) {
						membershipInfo.emplace_back(hostDataClusterMembership.at(i), i);
					}
					std::sort(membershipInfo.begin(), membershipInfo.end());
#endif

					// Update cluster centers
#if USE_OPTIMIZED == 1
					auto lastClusterCenter = UINT32_MAX;
					auto index = 0u;
					memset(hostEntryCount.get(), 0, hostEntryCount.bytes());
					for (auto& membership : membershipInfo) {
						hostSortedDataClusterMembership.at(index) = membership.second;
						++hostEntryCount.at(membership.first);
						if (lastClusterCenter != membership.first) {
							hostEntryPoints.at(membership.first) = index;
							lastClusterCenter = membership.first;
						}
						++index;
					}


					blockDim = dim3(32, 8);
					gridDim = dim3((clusters.rows + blockDim.x - 1) / blockDim.x,
								   (clusters.columns + blockDim.y - 1) / blockDim.y);

					gridDim.x = std::max(1u, gridDim.x / 2u);

					hostSortedDataClusterMembership.copyTo(dataClusterMembership);
					hostEntryCount.copyTo(deviceEntryCount);
					hostEntryPoints.copyTo(deviceEntryPoints);
					details::kernelUpdateClusterCenters<NumericType><<<gridDim, blockDim>>>(clusters.rows, clusters.columns, data.dense.values, data.dense.leadingDimension, clusters.dense.values, clusters.dense.leadingDimension, dataClusterMembership.get(), deviceEntryPoints.get(), deviceEntryCount.get());
					hostDataClusterMembership.copyTo(dataClusterMembership);
#else
					auto streamID = 0u;
					for (auto clusterID = 0u; clusterID < clusters.columns(); ++clusterID) {
						auto count = membershipBuckets.count(clusterID); 

						// Set a cuBLAS stream for this operation an choose next index
						CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, streams[streamID]));
						streamID = ++streamID % streams.size();

						// Does any membership exist?
						if (count == 0) {
							//std::cout << " [WARNING] k-Means: Empty cluster occurred in iteration " << iteration << ", choosing a new random cluster" << std::endl;

							// Set the vector to a random data entry
							auto randomIndex = std::uniform_int_distribution<unsigned>(0, data.columns() - 1)(generator);
							CUBLAS_CALL(wrapper::cublasXcopy(static_cast<int>(clusters.rows()), &data.at(0, randomIndex), 1, &clusters.at(0, clusterID), 1));
						} else if (count == 1) {
							// Just copy the data entry as cluster center
							CUBLAS_CALL(wrapper::cublasXcopy(static_cast<int>(clusters.rows()), &data.at(0, membershipBuckets.find(clusterID)->second), 1, &clusters.at(0, clusterID), 1));
						} else {
							// Get iterators of cluster
							auto it = membershipBuckets.lower_bound(clusterID);
							auto end = membershipBuckets.upper_bound(clusterID);

							// Clear cluster center
							auto alpha = NumericType(0.0);
							CUBLAS_CALL(wrapper::cublasXscal(static_cast<int>(clusters.rows()), &alpha, &clusters.at(0, clusterID), 1));

							// Add data entries to cluster center
							alpha = NumericType(1.0 / count);
							while (it != end) {
								CUBLAS_CALL(wrapper::cublasXaxpy(static_cast<int>(clusters.rows()), &alpha, &data.at(0, it->second), 1, &clusters.at(0, clusterID), 1));
								it++;
							}
						}
					}
#endif
				}
			} while (++iteration < maxiter && percentageChange > threshold);

			if (percentageChange > 0.0) {
				// Finally compute memberships of data entries
				auto membershipChangeCount = 0u;
				CUDA_CALL(cudaMemcpy(deviceMembershipChangeCount.get(), &membershipChangeCount, sizeof(unsigned), cudaMemcpyHostToDevice));

				auto blockDim = dim3(32, 4);
				auto gridDim = dim3((data.columns + blockDim.y - 1) / blockDim.y);
				details::kernelUpdateMembership<NumericType> << <gridDim, blockDim >> >(data.columns, data.rows, clusters.columns, data.dense.values, data.dense.leadingDimension, clusters.dense.values, clusters.dense.leadingDimension, dataClusterMembership.get(), deviceMembershipChangeCount.get());
			}
		}
	}

	void computeKMeans(const MatrixDescription<float>& data, const MatrixDescription<float>& clusters, DeviceMemory<unsigned>& dataClusterMembership, unsigned seed, unsigned maxiter /* = 100u */, double threshold /* = 0.05 */) {
		details::computeKMeans(data, clusters, dataClusterMembership, seed, maxiter, threshold);
	}

	void computeKMeans(const MatrixDescription<double>& data, const MatrixDescription<double>& clusters, DeviceMemory<unsigned>& dataClusterMembership, unsigned seed, unsigned maxiter /* = 100u */, double threshold /* = 0.05 */) {
		details::computeKMeans(data, clusters, dataClusterMembership, seed, maxiter, threshold);
	}
}