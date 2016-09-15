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

#include <algorithm>
#include <cmath>
#include <common/Matrix.h>
#include <common/Misc.h>
#include <common/Event.h>
#include <common/Stream.h>
#include <cuda_runtime_api.h>
#include <nmf/Algorithm.h>
#include <nmf/Context.h>
#include <nmf/FrobeniusResolver.h>

#include <curand.h>

namespace nmfgpu {
	/** 
	Error function: sqrt(trace(V^TV) - 2*trace(H^TS^TW^TV) + trace(SHH^TS^TW^TW))
	@tparam NumericType */
	template<typename NumericType>
	class AlgorithmNonSmoothNMF : public IAlgorithm {
		struct TemporaryData {
			DeviceMatrix<NumericType> deviceS;
			DeviceMatrix<NumericType> deviceRR;
			DeviceMatrix<NumericType> deviceRN;
			DeviceMatrix<NumericType> deviceMR;
			DeviceMatrix<NumericType> deviceRN2;
			DeviceMatrix<NumericType> deviceMR2;
			DeviceMatrix<NumericType> deviceRNMRMemory;
			DeviceMatrix<NumericType> deviceRNMRMemory2;
			DeviceMemory<NumericType> devicePartialSumsR;
			DeviceMemory<NumericType> devicePartialSumsN;
			HostMemory<NumericType> hostPartialSumsVTV;
			HostMemory<NumericType> hostPartialSumsHTWTV;
			HostMemory<NumericType> hostPartialSumsHHTWTW;
		};

		std::unique_ptr<NMFCoreMatrices<NumericType>> m_coreMatrices;
		std::unique_ptr<TemporaryData> m_temporaryData;

		NmfDescription<NumericType>& m_context;
		double m_theta;
		Stream m_streams[3];
		Event m_eventTraceFinished, m_eventMemcpyFinished;

		void computeMatrixH(bool computeError);
		void computeMatrixW(bool computeError);

	public:
		AlgorithmNonSmoothNMF(NmfDescription<NumericType>& desc, double theta)
			: IAlgorithm(desc.seed)
			, m_context(desc) 
			, m_theta(theta) { }

		virtual ResultType allocateMemory() override;
		virtual void deallocateMemory() override;

		virtual void computeIteration(bool computeError) override;
		virtual void storeFactorization() override;
		virtual void initialize() override;
		
		virtual const char* name() const override {
			return "non-smooth NMF";
		}
	};

	template<typename NumericType>
	ResultType AlgorithmNonSmoothNMF<NumericType>::allocateMemory() {
		m_coreMatrices = make_unique<NMFCoreMatrices<NumericType>>();
		m_temporaryData = make_unique<TemporaryData>();

		auto m = m_context.inputMatrix.rows;
		auto n = m_context.inputMatrix.columns;
		auto r = m_context.features;

		// Allocate device memory
		if (!m_coreMatrices->deviceW.allocate(m, r)
			|| !m_coreMatrices->deviceH.allocate(r, n)
			|| !m_coreMatrices->deviceV.allocate(m, n)
			|| !m_temporaryData->deviceS.allocate(r, r)
			|| !m_temporaryData->deviceRR.allocate(r, r)
			|| !m_temporaryData->deviceRNMRMemory.allocate(r, n, { { m, r } })
			|| !m_temporaryData->deviceRNMRMemory2.allocate(r, n, { { m, r } })
			|| !m_temporaryData->devicePartialSumsN.allocate(kernel::traceMultiplicationGetElementCount(n))
			|| !m_temporaryData->devicePartialSumsR.allocate(kernel::traceMultiplicationGetElementCount(r))) {
			return ResultType::ErrorNotEnoughDeviceMemory;
		}

		m_temporaryData->deviceRN = m_temporaryData->deviceRNMRMemory.as(MatrixDimension{ r, n });
		m_temporaryData->deviceMR = m_temporaryData->deviceRNMRMemory.as(MatrixDimension{ m, r });
		m_temporaryData->deviceRN2 = m_temporaryData->deviceRNMRMemory2.as(MatrixDimension{ r, n });
		m_temporaryData->deviceMR2 = m_temporaryData->deviceRNMRMemory2.as(MatrixDimension{ m, r });

		// Allocate host memory
		if (!m_temporaryData->hostPartialSumsVTV.allocate(n)
			|| !m_temporaryData->hostPartialSumsHTWTV.allocate(n)
			|| !m_temporaryData->hostPartialSumsHHTWTW.allocate(r)) {
			return ResultType::ErrorNotEnoughHostMemory;
		}

		// Compute trace(V^T*V)
		m_coreMatrices->deviceV.copyFrom(m_context.inputMatrix);
		m_coreMatrices->deviceV.traceMultiplication(true, m_coreMatrices->deviceV, m_temporaryData->devicePartialSumsN);

		auto& hpsVTV = m_temporaryData->hostPartialSumsVTV;
		auto& dpsVTV = m_temporaryData->devicePartialSumsN;
		dpsVTV.copyTo(hpsVTV);
		std::sort(&hpsVTV.get()[0], &hpsVTV.get()[hpsVTV.elements()]);

		// Initialize constant smoothing matrix
		auto offDiagValueS = static_cast<NumericType>(m_theta) / static_cast<NumericType>(m_context.features);
		auto diagValueS = (1.0 - static_cast<NumericType>(m_theta)) + offDiagValueS;
		m_temporaryData->deviceS.fill(offDiagValueS, diagValueS);

		return ResultType::Success;
	}
	
	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::deallocateMemory() {
		m_coreMatrices = nullptr;
		m_temporaryData = nullptr;
	}
	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::initialize() {
		m_context.seed = generateRandomNumber();
		auto strategy = InitializationStrategy<NumericType>::create(m_context, *m_coreMatrices);
		if (strategy != nullptr) {
			strategy->initializeMatrixW(m_coreMatrices->deviceW);
			strategy->initializeMatrixH(m_coreMatrices->deviceH);
		}

		if (m_context.useConstantBasisVectors) {
			m_coreMatrices->deviceW.copyFrom(m_context.outputMatrixW);
		}
	}

	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::computeIteration(bool computeError) {
		computeMatrixH(computeError);
		computeMatrixW(computeError);

		// Resolve frobenius norm
		if (computeError) {
			m_eventMemcpyFinished.synchronize();
			auto frobenius = resolveFrobenius(m_temporaryData->hostPartialSumsVTV, m_temporaryData->hostPartialSumsHTWTV, m_temporaryData->hostPartialSumsHHTWTW);
			auto rmsd = frobenius / std::sqrt(m_coreMatrices->deviceV.rows() * m_coreMatrices->deviceV.columns());

			setRemainingError(frobenius, rmsd);
		}
	}

	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::computeMatrixH(bool computeError) {
		m_temporaryData->deviceMR2 = m_coreMatrices->deviceW * m_temporaryData->deviceS;
		m_temporaryData->deviceRR = m_temporaryData->deviceMR2.transposed() * m_temporaryData->deviceMR2;
		m_temporaryData->deviceRN = m_temporaryData->deviceMR2.transposed() * m_coreMatrices->deviceV;
		m_temporaryData->deviceRN2 = m_temporaryData->deviceRR * m_coreMatrices->deviceH;

		m_coreMatrices->deviceH.multiplyDivide(m_temporaryData->deviceRN, m_temporaryData->deviceRN2, std::numeric_limits<NumericType>::epsilon());

		// If error should be computed then compute: trace(H^T*(S^T*W^T*V))
		if (computeError) {
			m_coreMatrices->deviceH.traceMultiplication(true, m_temporaryData->deviceRN, m_temporaryData->devicePartialSumsN);
			m_temporaryData->devicePartialSumsN.copyTo(m_temporaryData->hostPartialSumsHTWTV);
		}
	}

	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::computeMatrixW(bool computeError) {
		// If SSNMF is enabled and no error function should be computed then exit the update for matrix W
		if (!computeError && m_context.useConstantBasisVectors) {
			return;
		}

		m_temporaryData->deviceRN2 = m_temporaryData->deviceS * m_coreMatrices->deviceH;
		m_temporaryData->deviceRR = m_temporaryData->deviceRN2 * m_temporaryData->deviceRN2.transposed();

		// If error should be computed then compute: trace((SHH^TS^T)(W^TW))
		if (computeError) {
			auto viewRR2 = m_temporaryData->deviceRNMRMemory.as(MatrixDimension{ m_context.features, m_context.features });
			viewRR2 = m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceW;
			m_temporaryData->deviceRR.traceMultiplication(false, viewRR2, m_temporaryData->devicePartialSumsR);
			m_temporaryData->devicePartialSumsR.copyTo(m_temporaryData->hostPartialSumsHHTWTW);
			m_eventMemcpyFinished.record();

			// If SSNMF is enabled then exit the update for matrix W
			if (m_context.useConstantBasisVectors) {
				return;
			}
		}

		m_temporaryData->deviceMR = m_coreMatrices->deviceV * m_temporaryData->deviceRN2.transposed();
		m_temporaryData->deviceMR2 = m_coreMatrices->deviceW * m_temporaryData->deviceRR;

		m_coreMatrices->deviceW.multiplyDivide(m_temporaryData->deviceMR, m_temporaryData->deviceMR2, std::numeric_limits<NumericType>::epsilon());
		m_coreMatrices->deviceW.normalizeColumns();
	}

	template<typename NumericType>
	void AlgorithmNonSmoothNMF<NumericType>::storeFactorization() {
		m_temporaryData->deviceMR = m_coreMatrices->deviceW * m_temporaryData->deviceS; 
		m_temporaryData->deviceMR.copyTo(m_context.outputMatrixW);
		m_coreMatrices->deviceH.copyTo(m_context.outputMatrixH);
	}
}