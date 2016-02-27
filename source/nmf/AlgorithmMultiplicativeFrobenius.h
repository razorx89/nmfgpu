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
	@tparam NumericType */
	template<typename NumericType>
	class AlgorithmMultiplicativeFrobenius : public IAlgorithm {
		struct TemporaryData {
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
		Stream m_streams[3];
		Event m_eventTraceFinished, m_eventMemcpyFinished;

		void computeMatrixH(bool computeError);
		void computeMatrixW(bool computeError);

	public:
		AlgorithmMultiplicativeFrobenius(NmfDescription<NumericType>& desc)
			: IAlgorithm(desc.seed)
			, m_context(desc) { }

		virtual ResultType allocateMemory() override;
		virtual void deallocateMemory() override;

		virtual void computeIteration(bool computeError) override;
		virtual void storeFactorization() override;
		virtual void initialize() override;
		
		virtual const char* name() const override {
			return "Multiplicative Frobenius";
		}
	};

	template<typename NumericType>
	ResultType AlgorithmMultiplicativeFrobenius<NumericType>::allocateMemory() {
		m_coreMatrices = make_unique<NMFCoreMatrices<NumericType>>();
		m_temporaryData = make_unique<TemporaryData>();

		auto m = m_context.inputMatrix.rows;
		auto n = m_context.inputMatrix.columns;
		auto r = m_context.features;

		// Allocate device memory
		if (!m_coreMatrices->deviceW.allocate(m, r)
			|| !m_coreMatrices->deviceH.allocate(r, n)
			|| !m_coreMatrices->deviceV.allocate(m, n)
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
		cudaDeviceSynchronize();
		std::sort(&hpsVTV.get()[0], &hpsVTV.get()[hpsVTV.elements()]);

		return ResultType::Success;
	}
	
	template<typename NumericType>
	void AlgorithmMultiplicativeFrobenius<NumericType>::deallocateMemory() {
		m_coreMatrices = nullptr;
		m_temporaryData = nullptr;
	}
	template<typename NumericType>
	void AlgorithmMultiplicativeFrobenius<NumericType>::initialize() {
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
	void AlgorithmMultiplicativeFrobenius<NumericType>::computeIteration(bool computeError) {
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
	void AlgorithmMultiplicativeFrobenius<NumericType>::computeMatrixH(bool computeError) { 
		if (computeError) {
			// Compute W^T * W using a general matrix multiplication
			m_temporaryData->deviceRR = (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceW)
				.async(m_streams[0]);

			// Compute (W^T * W) * H using a general matrix multiplication
			m_temporaryData->deviceRN2 = (m_temporaryData->deviceRR * m_coreMatrices->deviceH)
				.async(m_streams[0]);
		} else {
			// Compute W^T * W using a symmetric rank update and store the result in the lower triangle
			m_temporaryData->deviceRR = (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceW)
				.async(m_streams[0])
				.syrk(FillType::Lower);

			// Compute (W^T * W) * H using a symmetric matrix multiplication
			m_temporaryData->deviceRN2 = (m_temporaryData->deviceRR * m_coreMatrices->deviceH)
				.async(m_streams[0])
				.symm(SideType::LeftSide, FillType::Lower);
		}

		// Compute W^T * V
		m_temporaryData->deviceRN = (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceV)
			.async(m_streams[1]);

		// Invoke multiplicative update kernel
		m_coreMatrices->deviceH.multiplyDivide(m_temporaryData->deviceRN, m_temporaryData->deviceRN2, std::numeric_limits<NumericType>::epsilon());

		// Compute trace(H^T*W^T*V)
		if (computeError) {
			m_coreMatrices->deviceH.traceMultiplication(true, m_temporaryData->deviceRN, m_temporaryData->devicePartialSumsN);
			m_temporaryData->devicePartialSumsN.copyTo(m_temporaryData->hostPartialSumsHTWTV, m_streams[2]);
		}
	}

	template<typename NumericType>
	void AlgorithmMultiplicativeFrobenius<NumericType>::computeMatrixW(bool computeError) {
		if (computeError) {
			// Copy W^T*W
			auto tmpRR = m_temporaryData->deviceRN2.as(m_temporaryData->deviceRR.dimension());
			m_temporaryData->deviceRR.copyTo(tmpRR/*, m_streams[0]*/);

			// Compute H * H^T using a general matrix multiplication
			m_temporaryData->deviceRR = (m_coreMatrices->deviceH * m_coreMatrices->deviceH.transposed())
				.async(m_streams[0]);

			// Compute trace(H*H^T*W^T*W)
			m_temporaryData->deviceRR.traceMultiplication(false, tmpRR, m_temporaryData->devicePartialSumsR, m_streams[0]);
			m_eventTraceFinished.record(m_streams[0]);
			m_streams[2].waitFor(m_eventTraceFinished);
			m_temporaryData->devicePartialSumsR.copyTo(m_temporaryData->hostPartialSumsHHTWTW, m_streams[2]);
			m_eventMemcpyFinished.record(m_streams[2]);
			
			if (m_context.useConstantBasisVectors) {
				return;
			}

			// Compute W * (H * H^T)
			m_temporaryData->deviceMR2 = (m_coreMatrices->deviceW * m_temporaryData->deviceRR)
				.async(m_streams[0]);
		} else {
			if (m_context.useConstantBasisVectors) {
				return;
			}

			// Compute H * H^T using a symmetric rank update and store the result in the upper triangle
			m_temporaryData->deviceRR = (m_coreMatrices->deviceH * m_coreMatrices->deviceH.transposed())
				.syrk(FillType::Upper);

			// Compute W * (H * H^T) using a symmetric matrix multiplication
			m_temporaryData->deviceMR2 = (m_coreMatrices->deviceW * m_temporaryData->deviceRR)
				.symm(SideType::RightSide, FillType::Upper);
		}

		// Compute V*H^T
		m_temporaryData->deviceMR = (m_coreMatrices->deviceV * m_coreMatrices->deviceH.transposed())
			.async(m_streams[1]);

		// Invoke multiplicative update kernel
		m_coreMatrices->deviceW.multiplyDivide(m_temporaryData->deviceMR, m_temporaryData->deviceMR2, std::numeric_limits<NumericType>::epsilon());

		// Normalize columns to unity
		m_coreMatrices->deviceW.normalizeColumns();
	}

	template<typename NumericType>
	void AlgorithmMultiplicativeFrobenius<NumericType>::storeFactorization() {
		m_coreMatrices->deviceW.copyTo(m_context.outputMatrixW);
		m_coreMatrices->deviceH.copyTo(m_context.outputMatrixH);
	}
}