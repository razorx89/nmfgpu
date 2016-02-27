#pragma once

namespace nmfgpu {
	template<typename NumericType>
	struct MatrixDescription;

	/** Computes the k-Means cluster centers for a given data matrix. The computation can be configured by a seed for initial cluster selection, 
	a maximum iteration count and an early exit if less than the specified threshold (percentage) of cluster centers are changing. 
	@param data Data matrix with attributes in rows and data entries in columns. 
	@param clusters Cluster matrix with attributes in rows and cluster centers in columns. 
	@param dataClusterMembership Contains after algorithm execution the indices of data entries for each cluster center. Size must be at least data.columns().
	@param seed Using different seeds leads to a different cluster selection at startup of the algorithm.
	@param maxiter The algorithm will stop the execution if the maximum iteration count is reached.
	@param threshold Percentage value which is used to stop the execution if too few cluster centers are changing. */
	void computeKMeans(const MatrixDescription<float>& data, const MatrixDescription<float>& clusters, DeviceMemory<unsigned>& dataClusterMembership, unsigned seed, unsigned maxiter = 100u, double threshold = 0.05);
	void computeKMeans(const MatrixDescription<double>& data, const MatrixDescription<double>& clusters, DeviceMemory<unsigned>& dataClusterMembership, unsigned seed, unsigned maxiter = 100u, double threshold = 0.05);
}