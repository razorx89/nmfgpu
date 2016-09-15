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

#include <common/Event.h>
#include <common/Matrix.h>
#include <common/Misc.h>
#include <common/Memory.h>
#include <common/Stream.h>
#include <init/InitializationStrategy.h>
#include <nmf/Algorithm.h>
#include <nmf/Context.h>

namespace nmfgpu {
	template<typename NumericType>
	class AlgorithmHoyerConstrainedAlternatingLeastSquares : public IAlgorithm {
		struct TemporaryData {
			DeviceMatrix<NumericType> deviceRR;
			DeviceMatrix<NumericType> deviceRR2;
			DeviceMatrix<NumericType> deviceMR;
			DeviceMemory<NumericType> devicePartialSumsR;
			DeviceMemory<NumericType> devicePartialSumsN;
			DeviceMemory<NumericType> deviceQRTau;
			DeviceMemory<NumericType> deviceQRWorkspace;
			DeviceMemory<int> deviceInfo;
			HostMatrix<NumericType> hostRR;
			HostMemory<NumericType> hostPartialSumsVTV;
			HostMemory<NumericType> hostPartialSumsWTVHT;
			HostMemory<NumericType> hostPartialSumsHHTWTW;
		};
		
		std::unique_ptr<NMFCoreMatrices<NumericType>> m_coreMatrices;
		std::unique_ptr<TemporaryData> m_temporaryData;

		NmfDescription<NumericType>& m_context;
		NumericType m_betaW, m_betaH;
		NumericType m_lambdaW, m_lambdaH;
		bool m_modeACLS;

		Stream m_streams[3];
		Event m_eventTraceFinished, m_eventMemcpyFinished;

	public:
		AlgorithmHoyerConstrainedAlternatingLeastSquares(NmfDescription<NumericType>& context, NumericType lambdaW, NumericType lambdaH)
			: IAlgorithm(context.seed)
			, m_context(context)
			, m_betaW(0.f)
			, m_betaH(0.f)
			, m_lambdaW(lambdaW)
			, m_lambdaH(lambdaH)
			, m_modeACLS(true) {
		}

		AlgorithmHoyerConstrainedAlternatingLeastSquares(NmfDescription<NumericType>& context,
														 NumericType lambdaW, NumericType lambdaH, 
														 NumericType alphaW, NumericType alphaH)
			: IAlgorithm(context.seed)
			, m_context(context)
			, m_lambdaW(lambdaW)
			, m_lambdaH(lambdaH)
			, m_modeACLS(false) {
			m_betaW = ((1 - alphaW) * std::sqrt(context.features) + alphaW);
			m_betaW *= m_betaW;
			m_betaH = ((1 - alphaH) * std::sqrt(context.features) + alphaH);
			m_betaH *= m_betaH;
		}

		virtual ResultType allocateMemory() override;
		virtual void deallocateMemory() override;

		virtual void computeIteration(bool computeError) override;
		virtual void storeFactorization() override;
		virtual void initialize() override;

		virtual const char* name() const override {
			if (m_modeACLS)
				return "Alternating Constrained Least Squares";
			else
				return "Alternating Hoyer Constrained Least Squares";
		}
	};

	template<typename NumericType>
	ResultType AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>::allocateMemory() {
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
			|| !m_temporaryData->deviceRR2.allocate(r, r)
			|| !m_temporaryData->deviceMR.allocate(m, r)
			|| !m_temporaryData->devicePartialSumsN.allocate(kernel::traceMultiplicationGetElementCount(n))
			|| !m_temporaryData->devicePartialSumsR.allocate(kernel::traceMultiplicationGetElementCount(r))
			|| !m_temporaryData->deviceQRTau.allocate(r)
			|| !m_temporaryData->deviceQRWorkspace.allocate(m_temporaryData->deviceRR.qrWorkspaceSize())
			|| !m_temporaryData->deviceInfo.allocate(1)) {
			return ResultType::ErrorNotEnoughDeviceMemory;
		}

		// Allocate host memory
		if (!m_temporaryData->hostPartialSumsVTV.allocate(n)
			|| !m_temporaryData->hostPartialSumsWTVHT.allocate(r)
			|| !m_temporaryData->hostPartialSumsHHTWTW.allocate(r)) {
			return ResultType::ErrorNotEnoughHostMemory;
		}

		// Compute trace(V^T*V)
		m_coreMatrices->deviceV.copyFrom(m_context.inputMatrix);
		m_coreMatrices->deviceV.traceMultiplication(true, m_coreMatrices->deviceV, m_temporaryData->devicePartialSumsN);

		auto& hpsVTV = m_temporaryData->hostPartialSumsVTV;
		auto& dpsVTV = m_temporaryData->devicePartialSumsN;
		CUDA_CALL(cudaMemcpy(hpsVTV.get(), dpsVTV.get(), sizeof(NumericType) * hpsVTV.elements(), cudaMemcpyDeviceToHost));
		std::sort(&hpsVTV.get()[0], &hpsVTV.get()[hpsVTV.elements()]);

		return ResultType::Success;
	}

	template<typename NumericType>
	void AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>::deallocateMemory() {
		m_coreMatrices = nullptr;
		m_temporaryData = nullptr;
	}

	template<typename NumericType>
	void AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>::storeFactorization() {
		m_coreMatrices->deviceW.copyTo(m_context.outputMatrixW);
		m_coreMatrices->deviceH.copyTo(m_context.outputMatrixH);
	}

	template<typename NumericType>
	void AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>::initialize() {
		m_context.seed = generateRandomNumber();
		auto strategy = InitializationStrategy<NumericType>::create(m_context, *m_coreMatrices);
		if (strategy != nullptr) {
			strategy->initializeMatrixW(m_coreMatrices->deviceW);
		}

		if (m_context.useConstantBasisVectors) {
			m_coreMatrices->deviceW.copyFrom(m_context.outputMatrixW);
		}
	}

	template<typename NumericType>
	void AlgorithmHoyerConstrainedAlternatingLeastSquares<NumericType>::computeIteration(bool computeError) {
		// Compute matrix H -------------------------------------

		if (computeError) {
			// Compute W^T*W using a general matrix multiplication
			m_temporaryData->deviceRR = (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceW)
				.async(m_streams[0]);

			// Save W^T*W for later trace computation
			m_temporaryData->deviceRR.copyTo(m_temporaryData->deviceRR2, m_streams[0]);

			// Add constraints to the covariance matrix
			if (m_modeACLS)
				m_temporaryData->deviceRR.addConstant(NumericType(0.0), m_lambdaH, m_streams[0]);
			else
				m_temporaryData->deviceRR.addConstant(-m_lambdaH, m_lambdaH * m_betaH - m_lambdaH, m_streams[0]);
		} else {
			// Initialize the covariance matrix with constraints
			if (m_modeACLS)
				m_temporaryData->deviceRR.fill(NumericType(0.0), m_lambdaH, m_streams[0]);
			else
				m_temporaryData->deviceRR.fill(-m_lambdaH, m_lambdaH * m_betaH - m_lambdaH, m_streams[0]);

			// Compute W^T*W using a general matrix multiplication and add it to the constraints
			m_temporaryData->deviceRR += (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceW)
				.async(m_streams[0]);
		}

		// Compute qr(W^T*W)
		m_temporaryData->deviceRR.qr(m_temporaryData->deviceQRTau, m_temporaryData->deviceQRWorkspace, m_temporaryData->deviceInfo, m_streams[0]);

		// Compute W^T*V using a general matrix multiplication
		m_coreMatrices->deviceH = (m_coreMatrices->deviceW.transposed() * m_coreMatrices->deviceV)
			.async(m_streams[1]);

		// Compute Q^T*W^T*V
		m_coreMatrices->deviceH.ormqr(SideType::LeftSide, true, m_temporaryData->deviceRR, m_temporaryData->deviceQRTau, m_temporaryData->deviceQRWorkspace, m_temporaryData->deviceInfo);

		// Solve least squares system
		DeviceMatrix<NumericType>::solveLinearEquationSystem(SideType::LeftSide, FillType::Upper, m_temporaryData->deviceRR, false, m_coreMatrices->deviceH);

		// Set all negative values to zero
		m_coreMatrices->deviceH.setNegativeToZero();

		// Compute matrix W -------------------------------------

		if (computeError) {
			// Compute H*H^T using a general matrix multiplication
			m_temporaryData->deviceRR = (m_coreMatrices->deviceH * m_coreMatrices->deviceH.transposed())
				.async(m_streams[0]);

			// Compute trace(H*H^T*W^T*W)
			m_temporaryData->deviceRR.traceMultiplication(false, m_temporaryData->deviceRR2, m_temporaryData->devicePartialSumsR, m_streams[0]);
			m_temporaryData->devicePartialSumsR.copyTo(m_temporaryData->hostPartialSumsHHTWTW, m_streams[0]); // FIXME m_streams[2]!

			if (!m_context.useConstantBasisVectors) {
				// Add constraints to the covariance matrix
				if (m_modeACLS)
					m_temporaryData->deviceRR.addConstant(NumericType(0.0), m_lambdaW, m_streams[0]);
				else
					m_temporaryData->deviceRR.addConstant(-m_lambdaW, m_lambdaW * m_betaW - m_lambdaW, m_streams[0]);
			}
		} else {
			if (!m_context.useConstantBasisVectors) {
				// Initialize the covariance matrix with constraints
				if (m_modeACLS)
					m_temporaryData->deviceRR.fill(NumericType(0.0), m_lambdaW, m_streams[0]);
				else
					m_temporaryData->deviceRR.fill(-m_lambdaW, m_lambdaW * m_betaW - m_lambdaW, m_streams[0]);

				// Compute H*H^T using a general matrix multiplication
				m_temporaryData->deviceRR += (m_coreMatrices->deviceH * m_coreMatrices->deviceH.transposed())
					.async(m_streams[0]);
			}
		}

		if (!m_context.useConstantBasisVectors) {
			// Compute qr(H*H^T)
			m_temporaryData->deviceRR.qr(m_temporaryData->deviceQRTau, m_temporaryData->deviceQRWorkspace, m_temporaryData->deviceInfo, m_streams[0]);
		}

		// Save W for later trace computation
		if (computeError) {
			m_coreMatrices->deviceW.copyTo(m_temporaryData->deviceMR, m_streams[1]);
		}

		if (!m_context.useConstantBasisVectors) {
			// Compute V*H^T using a general matrix multiplication
			m_coreMatrices->deviceW = (m_coreMatrices->deviceV * m_coreMatrices->deviceH.transposed())
				.async(m_streams[1]);
		}

		// Compute trace(W^T*V*H^T)
		if (computeError) {
			m_temporaryData->deviceMR.traceMultiplication(true, m_coreMatrices->deviceW, m_temporaryData->devicePartialSumsN, m_streams[1]);
			m_eventTraceFinished.record(m_streams[1]);
			m_streams[2].waitFor(m_eventTraceFinished);
			m_temporaryData->devicePartialSumsN.copyTo(m_temporaryData->hostPartialSumsWTVHT, m_streams[2]);
			m_eventMemcpyFinished.record(m_streams[2]);
		}

		if (!m_context.useConstantBasisVectors) {
			// Compute V*H^T*Q
			m_coreMatrices->deviceW.ormqr(SideType::RightSide, false, m_temporaryData->deviceRR, m_temporaryData->deviceQRTau, m_temporaryData->deviceQRWorkspace, m_temporaryData->deviceInfo);

			// Solve least squares system
			DeviceMatrix<NumericType>::solveLinearEquationSystem(SideType::RightSide, FillType::Upper, m_temporaryData->deviceRR, true, m_coreMatrices->deviceW);

			// Set all negative values to zero
			m_coreMatrices->deviceW.setNegativeToZero();

			// Normalize columns to unity
			m_coreMatrices->deviceW.normalizeColumns();
		}


		// Resolve frobenius norm
		if (computeError) {
			m_eventMemcpyFinished.synchronize();
			auto frobenius = resolveFrobenius(m_temporaryData->hostPartialSumsVTV, m_temporaryData->hostPartialSumsWTVHT, m_temporaryData->hostPartialSumsHHTWTW);
			auto rmsd = frobenius / std::sqrt(m_coreMatrices->deviceV.rows() * m_coreMatrices->deviceV.columns());

			setRemainingError(frobenius, rmsd);
		}
	}
}