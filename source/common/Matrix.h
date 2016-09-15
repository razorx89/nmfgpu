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
#include <assert.h>
#include <common/Interface.h>
#include <common/Logging.h>
#include <common/MatrixDescription.h>
#include <common/Memory.h>
#include <common/Wrapper.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <nmf/KernelFillMatrix.h>
#include <nmf/KernelMakeNonNegative.h>
#include <nmf/KernelMultiplyDivide.h>
#include <nmf/KernelNormalizeColumns.h>
#include <nmf/KernelTraceMultiplication.h>

namespace nmfgpu {
	// Forward declarations
	template<typename NumericType>
	class HostMatrix;
	template<typename NumericType>
	class DeviceMatrix;

	struct MatrixDimension {
		std::size_t rows;
		std::size_t columns;
	};

	bool operator == (const MatrixDimension& left, const MatrixDimension& right);
	bool operator != (const MatrixDimension& left, const MatrixDimension& right);


	template<typename NumericType>
	class Matrix {
	protected:
		std::shared_ptr<IMemory> m_memory;

		/** Row count of the matrix. */
		size_t m_rows;

		/** Column count of the matrix. */
		size_t m_columns;

		/** Leading dimension of the matrix. */
		size_t m_leadingDimension;

		/** Initializes the memory with leadingDimension * columns elements and stores the dimension of the matrix.
		@param rows Row count of the matrix.
		@param columns Column count of the matrix.
		@param leadingDimension Leading dimension of the matrix. */
		Matrix(std::shared_ptr<IMemory> memory, size_t rows, size_t columns, size_t leadingDimension)
			: m_memory(std::move(memory)) 
			, m_rows(rows)
			, m_columns(columns)
			, m_leadingDimension(leadingDimension) { }

	public:
		size_t elements() const {
			return m_leadingDimension * m_rows;
		}

		size_t rows() const {
			return m_rows;
		}

		size_t columns() const {
			return m_columns;
		}

		MatrixDimension dimension() const {
			return MatrixDimension{ m_rows, m_columns };
		}

		size_t leadingDimension() const {
			return m_leadingDimension;
		}

		NumericType& at(size_t row, size_t column) const {
			// Do some range checking 
			if (row >= rows() || column >= columns()) {
				throw std::out_of_range("row >= rows() || column >= columns()");
			}

			// Reference to the element, padded by the leading dimension
			return get()[column * leadingDimension() + row];
		}

		NumericType* get() const {
			assert(m_memory != nullptr);
			return reinterpret_cast<NumericType*>(m_memory->raw());
		}
		
		virtual void copyTo(HostMatrix<NumericType>& hostMatrix, cudaStream_t stream = nullptr) const = 0;
		virtual void copyTo(DeviceMatrix<NumericType>& deviceMatrix, cudaStream_t stream = nullptr) const = 0;
	};

	namespace details {	
		/** Generic copy function to do a 2D copy of the matrix data. Type traits ensure that both memory types must have the same UnderlyingDataType.
		@tparam MemoryType Specifies the memory type for both matrices. */
		template<typename NumericType>
		void copyMatrixToMatrix(Matrix<NumericType>& matrixDst, const Matrix<NumericType>& matrixSrc, cudaMemcpyKind kind, cudaStream_t stream) {
			CUDA_CALL(cudaMemcpy2DAsync(matrixDst.get(), 
										matrixDst.leadingDimension() * sizeof(NumericType),
										matrixSrc.get(), 
										matrixSrc.leadingDimension() * sizeof(NumericType),
										matrixSrc.rows() * sizeof(NumericType),
										matrixSrc.columns(), 
										kind, 
										stream));
		}

		template<typename NumericType>
		void copyMatrixToMatrix(Matrix<NumericType>& matrixDst, const MatrixDescription<NumericType>& matrixSrc, cudaMemcpyKind kind, cudaStream_t stream) {
			if (matrixSrc.format == StorageFormat::Dense) {
				CUDA_CALL(cudaMemcpy2DAsync(matrixDst.get(),
											matrixDst.leadingDimension() * sizeof(NumericType),
											matrixSrc.dense.values,
											matrixSrc.dense.leadingDimension * sizeof(NumericType),
											matrixSrc.rows * sizeof(NumericType),
											matrixSrc.columns,
											kind,
											stream));
			} else if (kind == cudaMemcpyHostToDevice) {
				if (matrixSrc.format == StorageFormat::CSR) {
					DeviceMemory<NumericType> values(matrixSrc.csr.nnz);
					DeviceMemory<int> rowPtr(matrixSrc.rows + 1);
					DeviceMemory<int> colInd(matrixSrc.csr.nnz);

					CUDA_CALL(cudaMemcpyAsync(values.get(), matrixSrc.csr.values, values.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(rowPtr.get(), matrixSrc.csr.rowPtr, rowPtr.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(colInd.get(), matrixSrc.csr.columnIndices, colInd.bytes(), cudaMemcpyHostToDevice, stream));
					
					cusparseMatDescr_t matDescr;
					CUSPARSE_CALL(cusparseCreateMatDescr(&matDescr));
					CUSPARSE_CALL(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
					CUSPARSE_CALL(cusparseSetMatIndexBase(matDescr, matrixSrc.csr.base == IndexBase::Zero ? 
																	CUSPARSE_INDEX_BASE_ZERO : 
																	CUSPARSE_INDEX_BASE_ONE));
					CUSPARSE_CALL(wrapper::cusparseXcsr2dense(matrixSrc.rows,
						matrixSrc.columns,
						matDescr,
						values.get(),
						rowPtr.get(),
						colInd.get(),
						matrixDst.get(),
						matrixDst.leadingDimension()));
					CUDA_CALL(cudaStreamSynchronize(stream));

					CUSPARSE_CALL(cusparseDestroyMatDescr(matDescr));
				} else if (matrixSrc.format == StorageFormat::CSC) {
					DeviceMemory<NumericType> values(matrixSrc.csc.nnz);
					DeviceMemory<int> colPtr(matrixSrc.columns + 1);
					DeviceMemory<int> rowInd(matrixSrc.csc.nnz);

					CUDA_CALL(cudaMemcpyAsync(values.get(), matrixSrc.csc.values, values.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(colPtr.get(), matrixSrc.csc.columnPtr, colPtr.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(rowInd.get(), matrixSrc.csc.rowIndices, rowInd.bytes(), cudaMemcpyHostToDevice, stream));

					cusparseMatDescr_t matDescr;
					CUSPARSE_CALL(cusparseCreateMatDescr(&matDescr));
					CUSPARSE_CALL(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
					CUSPARSE_CALL(cusparseSetMatIndexBase(matDescr, matrixSrc.csc.base == IndexBase::Zero ? 
																	CUSPARSE_INDEX_BASE_ZERO : 
																	CUSPARSE_INDEX_BASE_ONE));
					CUSPARSE_CALL(wrapper::cusparseXcsc2dense(matrixSrc.rows,
						matrixSrc.columns,
						matDescr,
						values.get(),
						rowInd.get(),
						colPtr.get(),
						matrixDst.get(),
						matrixDst.leadingDimension()));
					CUSPARSE_CALL(cusparseDestroyMatDescr(matDescr));

					CUDA_CALL(cudaStreamSynchronize(stream));
				} else if (matrixSrc.format == StorageFormat::COO) {
					DeviceMemory<NumericType> values(matrixSrc.coo.nnz);
					DeviceMemory<int> rowInd(matrixSrc.coo.nnz);
					DeviceMemory<int> rowPtr(matrixSrc.rows + 1);
					DeviceMemory<int> colInd(matrixSrc.coo.nnz);

					CUDA_CALL(cudaMemcpyAsync(values.get(), matrixSrc.coo.values, values.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(rowInd.get(), matrixSrc.coo.rowIndices, rowInd.bytes(), cudaMemcpyHostToDevice, stream));
					CUDA_CALL(cudaMemcpyAsync(colInd.get(), matrixSrc.coo.columnIndices, colInd.bytes(), cudaMemcpyHostToDevice, stream));

					// Convert row indices to compressed format (CSR)
					CUSPARSE_CALL(cusparseXcoo2csr(g_context->cusparseHandle, rowInd.get(), matrixSrc.coo.nnz, matrixSrc.rows, rowPtr.get(), CUSPARSE_INDEX_BASE_ZERO));

					// Create cuSPARSE matrix descriptor
					cusparseMatDescr_t matDescr;
					CUSPARSE_CALL(cusparseCreateMatDescr(&matDescr));
					CUSPARSE_CALL(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
					CUSPARSE_CALL(cusparseSetMatIndexBase(matDescr, matrixSrc.coo.base == IndexBase::Zero ? 
																	CUSPARSE_INDEX_BASE_ZERO : 
																	CUSPARSE_INDEX_BASE_ONE));

					// Convert from CSR to Dense
					CUSPARSE_CALL(wrapper::cusparseXcsr2dense(matrixSrc.rows,
						matrixSrc.columns,
						matDescr,
						values.get(),
						rowPtr.get(),
						colInd.get(),
						matrixDst.get(),
						matrixDst.leadingDimension()));
					
					// Delete descriptor and wait for stream to finish before the memory will get deallocated 
					CUSPARSE_CALL(cusparseDestroyMatDescr(matDescr));
					CUDA_CALL(cudaStreamSynchronize(stream));
				}
			} else {
				Logging::instance().error().print("[ERROR] Unsupported memory copy operation for sparse source matrix!").lineFeed();
			}
		}

		template<typename NumericType>
		void copyMatrixToMatrix(const MatrixDescription<NumericType>& matrixDst, const Matrix<NumericType>& matrixSrc, cudaMemcpyKind kind, cudaStream_t stream) {
			if (matrixDst.format != StorageFormat::Dense) {
				Logging::instance().error()
					.print("[ERROR] " NMFGPU_FILE_LINE_PREFIX ": Copy operation can only be performed between dense matrices!").lineFeed();
				return;
			}

			CUDA_CALL(cudaMemcpy2DAsync(matrixDst.dense.values,
				matrixDst.dense.leadingDimension * sizeof(NumericType),
				matrixSrc.get(),
				matrixSrc.leadingDimension() * sizeof(NumericType),
				matrixSrc.rows() * sizeof(NumericType),
				matrixSrc.columns(),
				kind,
				stream));
		}
	}
	
	template<typename NumericType>
	class HostMatrix : public Matrix<NumericType> {

	public:
		HostMatrix()
			: Matrix<NumericType>(nullptr, 0, 0, 0) { }

		/** Creates a new host matrix with specified dimension (rows x columns). No adjustment of the leading dimension will be done.
		@param rows Row count of the matrix. 
		@param columns Column count of the matrix. */
		HostMatrix(size_t rows, size_t columns)
			: Matrix<NumericType>(std::make_shared<HostMemory<NumericType>>(rows * columns), rows, columns, rows) { }

		bool allocate(size_t rows, size_t columns) {
			Matrix<NumericType>::m_leadingDimension = rows;
			Matrix<NumericType>::m_rows = rows;
			Matrix<NumericType>::m_columns = columns;
			Matrix<NumericType>::m_memory = std::make_shared<HostMemory<NumericType>>(Matrix<NumericType>::m_leadingDimension * columns);
			return Matrix<NumericType>::m_memory != nullptr;
		}

		/** Copies the host matrix to the specified host matrix. Therefore a host to host copy will be performed. 
		@param hostMatrix The destination host matrix which should contain the data from this matrix. 
		@param stream A CUDA stream to enable parallel copy mechanisms. */
		virtual void copyTo(HostMatrix<NumericType>& hostMatrix, cudaStream_t stream = nullptr) const override {
			details::copyMatrixToMatrix(hostMatrix, *this, cudaMemcpyHostToHost, stream);
		}

		/** Copies the host matrix to the specified device matrix. Therefore a host to device copy will be performed.
		@param deviceMatrix The destination device matrix which should contain the data from this matrix. 
		@param stream A CUDA stream to enable parallel copy mechanisms. */
		virtual void copyTo(DeviceMatrix<NumericType>& deviceMatrix, cudaStream_t stream = nullptr) const override {
			details::copyMatrixToMatrix(deviceMatrix, *this, cudaMemcpyHostToDevice, stream);
		}
	};
	
	enum class FillType {
		Full,
		Lower,
		Upper
	};

	enum class SideType {
		LeftSide,
		RightSide,
	};

	namespace details {
		template<typename NumericType>
		struct TransposedDeviceMatrix {
			DeviceMatrix<NumericType> matrix;
		};

		template<typename NumericType, typename EnableIf = void>
		class DeferredMultiplication;

		template<typename NumericType>
		class DeferredMultiplication<NumericType, typename std::enable_if<std::is_floating_point<NumericType>::value>::type> {
			DeviceMatrix<NumericType> m_left;
			bool m_transposedLeft;

			DeviceMatrix<NumericType> m_right;
			bool m_transposedRight;

			bool m_useSymmetricMultiplication{ false };
			SideType m_symmSide { SideType::LeftSide };
			FillType m_fillType{ FillType::Full };

			cudaStream_t m_stream{ nullptr };

			NumericType m_scaleMultiplication{ 1 };

			bool m_useSymmetricRankUpdate{ false };

			void doSymmetricRankUpdate(DeviceMatrix<NumericType>& target, NumericType alpha, NumericType beta) {
				CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, m_stream));
				CUBLAS_CALL(wrapper::cublasXsyrk(m_fillType == FillType::Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
					m_transposedLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
					static_cast<int>(m_transposedLeft ? m_left.columns() : m_left.rows()),
					static_cast<int>(m_transposedLeft ? m_left.rows() : m_left.columns()),
					&alpha,
					m_left.get(),
					static_cast<int>(m_left.leadingDimension()),
					&beta,
					target.get(),
					static_cast<int>(target.leadingDimension())));
			}

			void doSymmetricMatrixMultiplication(DeviceMatrix<NumericType>& target, NumericType alpha, NumericType beta) {
				CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, m_stream));
				CUBLAS_CALL(wrapper::cublasXsymm(m_symmSide == SideType::LeftSide ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
					m_fillType == FillType::Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
					static_cast<int>(target.rows()),
					static_cast<int>(target.columns()),
					&alpha,
					m_symmSide == SideType::LeftSide ? m_left.get() : m_right.get(),
					static_cast<int>(m_symmSide == SideType::LeftSide ? m_left.leadingDimension() : m_right.leadingDimension()),
					m_symmSide == SideType::LeftSide ? m_right.get() : m_left.get(),
					static_cast<int>(m_symmSide == SideType::LeftSide ? m_right.leadingDimension() : m_left.leadingDimension()),
					&beta,
					target.get(),
					static_cast<int>(target.leadingDimension())));
			}

			void doGeneralMatrixMultiplication(DeviceMatrix<NumericType>& target, NumericType alpha, NumericType beta) {
				CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, m_stream));
				CUBLAS_CALL(wrapper::cublasXgemm(m_transposedLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
					m_transposedRight ? CUBLAS_OP_T : CUBLAS_OP_N,
					static_cast<int>(m_transposedLeft ? m_left.columns() : m_left.rows()),
					static_cast<int>(m_transposedRight ? m_right.rows() : m_right.columns()),
					static_cast<int>(m_transposedLeft ? m_left.rows() : m_left.columns()),
					&alpha,
					m_left.get(),
					static_cast<int>(m_left.leadingDimension()),
					m_right.get(),
					static_cast<int>(m_right.leadingDimension()),
					&beta,
					target.get(),
					static_cast<int>(target.leadingDimension())));
			}

			void verify(const DeviceMatrix<NumericType>& target) {
				// Check if this memory is not included in operands
				if (target.get() == m_left.get() || target.get() == m_right.get()) {
					throw std::invalid_argument("multiplication target must be different from operands");
				}
			}

		public:
			DeferredMultiplication(const DeviceMatrix<NumericType>& left, bool transposeLeft, const DeviceMatrix<NumericType>& right, bool transposeRight)
				: m_left(left)
				, m_transposedLeft(transposeLeft)
				, m_right(right)
				, m_transposedRight(transposeRight)
				, m_stream(nullptr) { }

			DeferredMultiplication<NumericType>& async(cudaStream_t stream) {
				m_stream = stream;
				return *this;
			}

			DeferredMultiplication<NumericType>& scale(NumericType value) {
				m_scaleMultiplication = value;
				return *this;
			}

			DeferredMultiplication<NumericType>& symm(SideType side, FillType fill) {
				if (m_transposedLeft || m_transposedRight) {
					throw std::logic_error("Cannot use symmetric multiplication with transposed matrices");
				}

				m_symmSide = side;
				m_fillType = fill;
				m_useSymmetricMultiplication = true;
				return *this;
			}

			DeferredMultiplication<NumericType>& syrk(FillType fill) {
				if (m_left.get() != m_right.get()) {
					throw std::logic_error("Cannot use symmetric rank update with two different matrices!");
				}

				if (m_transposedLeft == m_transposedRight) {
					throw std::logic_error("Cannot use symmetric rank update with AxA or A^TxA^T multiplication!");
				}

				m_useSymmetricRankUpdate = true;
				m_fillType = fill;
				return *this;
			}

			void evaluate(DeviceMatrix<NumericType>& target, bool addToTarget) {
				verify(target);

				// Execute the multiplication, optimize if possible
				auto alpha = m_scaleMultiplication;
				auto beta = addToTarget ? NumericType(1.f) : NumericType(0.f);
				if (m_useSymmetricRankUpdate) {
					doSymmetricRankUpdate(target, alpha, beta);
				} else if (m_useSymmetricMultiplication) {
					doSymmetricMatrixMultiplication(target, alpha, beta);
				} else {
					doGeneralMatrixMultiplication(target, alpha, beta);
				}
			}
		};
	}

	template<typename NumericType>
	class DeviceMatrix : public Matrix<NumericType> {
		/** Computes the leading dimension, so that it will be padded to a multiple of 32.
		@param dimension Input dimension which might need additional padding.
		@returns Output dimension which is padded to a multiple of 32. */
		static size_t computeLeadingDimension(size_t dimension) {
			return (dimension + 31u) / 32u * 32u;
		}

		DeviceMatrix(std::shared_ptr<IMemory> memory, std::size_t rows, std::size_t columns, std::size_t ld)
			: Matrix<NumericType>(std::move(memory), rows, columns, ld) { }
	public:
		DeviceMatrix()
			: Matrix<NumericType>(nullptr, 0, 0, 0) { }

		MatrixDescription<NumericType> description() const { 
			auto desc = MatrixDescription<NumericType>();
			desc.format = StorageFormat::Dense;
			desc.rows = unsigned(Matrix<NumericType>::rows());
			desc.columns = unsigned(Matrix<NumericType>::columns());
			desc.dense.values = Matrix<NumericType>::get();
			desc.dense.leadingDimension = unsigned(Matrix<NumericType>::leadingDimension());
			return desc;
		}

		/** Creates a new device matrix with specified dimension (rows x columns), but adjusts the leading
		dimension to be a multiple of 32. This increases the cache performance of CUDA devices significantly.
		@param rows Row count of the matrix. 
		@param columns Column count of the matrix. */
		DeviceMatrix(size_t rows, size_t columns) 
			: Matrix<NumericType>(std::make_shared<DeviceMemory<NumericType>>(columns * DeviceMatrix::computeLeadingDimension(rows)), rows, columns, DeviceMatrix::computeLeadingDimension(rows)) { }

		bool allocate(size_t rows, size_t columns, std::initializer_list<MatrixDimension> compatibleDimensions = { }) {
			Matrix<NumericType>::m_leadingDimension = DeviceMatrix::computeLeadingDimension(rows);
			Matrix<NumericType>::m_rows = rows;
			Matrix<NumericType>::m_columns = columns;

			auto maxElements = Matrix<NumericType>::m_leadingDimension * columns;
			for (auto dimension : compatibleDimensions) {
				maxElements = std::max(DeviceMatrix::computeLeadingDimension(dimension.rows) * dimension.columns, maxElements);
			}

			Matrix<NumericType>::m_memory = std::make_shared<DeviceMemory<NumericType>>(maxElements);

			return Matrix<NumericType>::m_memory != nullptr;
		}

		DeviceMatrix as(MatrixDimension newDimension) {
			auto ld = DeviceMatrix::computeLeadingDimension(newDimension.rows);
			if (ld * newDimension.columns > Matrix<NumericType>::m_memory->elements()) {
				throw std::invalid_argument("Memory cannot be interpreted with this dimension!");
			}

			return DeviceMatrix(Matrix<NumericType>::m_memory, newDimension.rows, newDimension.columns, ld);
		}

		/** Copies the device matrix to the specified host matrix. Therefore a device to host copy will be performed.
		@param hostMatrix The destination host matrix which should contain the data from this matrix. 
		@param stream A CUDA stream to enable parallel copy mechanisms. */
		virtual void copyTo(HostMatrix<NumericType>& hostMatrix, cudaStream_t stream = nullptr) const override {
			details::copyMatrixToMatrix(hostMatrix, *this, cudaMemcpyDeviceToHost, stream);
		}

		/** Copies the device matrix to the specified device matrix. Therefore a device to device copy will be performed.
		@param deviceMatrix The destination device matrix which should contain the data from this matrix. 
		@param stream A CUDA stream to enable parallel copy mechanisms. */
		virtual void copyTo(DeviceMatrix<NumericType>& deviceMatrix, cudaStream_t stream = nullptr) const override {
			details::copyMatrixToMatrix(deviceMatrix, *this, cudaMemcpyDeviceToDevice, stream);
		}

		void copyFrom(const MatrixDescription<NumericType>& hostMatrix, cudaStream_t stream = nullptr) {
			details::copyMatrixToMatrix(*this, hostMatrix, cudaMemcpyHostToDevice, stream);
		}

		void copyTo(const MatrixDescription<NumericType>& hostMatrix, cudaStream_t stream = nullptr) const {
			details::copyMatrixToMatrix(hostMatrix, *this, cudaMemcpyDeviceToHost, stream);
		}

		details::TransposedDeviceMatrix<NumericType> transposed() const {
			return details::TransposedDeviceMatrix<NumericType> {*this};
		}

		void fill(NumericType value, cudaStream_t stream = nullptr) {
			fill(value, value, stream);
		}

		void fill(NumericType offDiagValue, NumericType diagValue, cudaStream_t stream = nullptr) {
			kernel::fillMatrix(description(), diagValue, offDiagValue, stream);
		}

		void addConstant(NumericType value, cudaStream_t stream = nullptr) {
			addConstant(value, value, stream);
		}

		void addConstant(NumericType offDiagValue, NumericType diagValue, cudaStream_t stream = nullptr) {
			kernel::addConstantToMatrix(description(), diagValue, offDiagValue, stream);
		}

		DeviceMatrix<NumericType>& operator = (details::DeferredMultiplication<NumericType> multiplication) {
			multiplication.evaluate(*this, false);
			return *this;
		}

		DeviceMatrix<NumericType>& operator += (details::DeferredMultiplication<NumericType> multiplication) {
			multiplication.evaluate(*this, true);
			return *this;
		}

		void multiplyDivide(const DeviceMatrix<NumericType>& numerator, const DeviceMatrix& denominator, NumericType epsilon, cudaStream_t stream = nullptr) {
			kernel::multiplyDivide(description(), numerator.description(), denominator.description(), epsilon, stream);
		}

		void normalizeColumns(cudaStream_t stream = nullptr) {
			kernel::normalizeColumns(description(), stream);
		}

		void traceMultiplication(bool transposeThis, const DeviceMatrix<NumericType>& right, DeviceMemory<NumericType>& partialSums, cudaStream_t stream = nullptr) const {
			kernel::traceMultiplication(transposeThis, description(), right.description(), partialSums, stream);
		}

		void qr(DeviceMemory<NumericType>& tau, DeviceMemory<NumericType>& workspace, DeviceMemory<int>& devInfo, cudaStream_t stream = nullptr) {
			CUSOLVER_CALL(cusolverDnSetStream(g_context->cusolverDnHandle, stream));
			CUSOLVER_CALL(wrapper::cusolverDnXgeqrf(static_cast<int>(Matrix<NumericType>::rows()), 
													static_cast<int>(Matrix<NumericType>::columns()), 
													Matrix<NumericType>::get(), 
													static_cast<int>(Matrix<NumericType>::leadingDimension()), 
													tau.get(), 
													workspace.get(), 
													static_cast<int>(workspace.elements()), 
													devInfo.get()));
		}

		int qrWorkspaceSize() const {
			int lwork;
			CUSOLVER_CALL(wrapper::cusolverDnXgeqrf_bufferSize(static_cast<int>(Matrix<NumericType>::rows()), 
															   static_cast<int>(Matrix<NumericType>::columns()), 
															   Matrix<NumericType>::get(), 
															   static_cast<int>(Matrix<NumericType>::leadingDimension()), 
															   &lwork));
			return lwork;
		}

		void ormqr(SideType sideOfMultiplication, bool transposeQ, const DeviceMatrix<NumericType>& matrixQR, const DeviceMemory<NumericType>& tau, const DeviceMemory<NumericType>& workspace, DeviceMemory<int>& devInfo, cudaStream_t stream = nullptr) {
			CUSOLVER_CALL(cusolverDnSetStream(g_context->cusolverDnHandle, stream));
			CUSOLVER_CALL(wrapper::cusolverDnXormqr(sideOfMultiplication == SideType::LeftSide ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
													transposeQ ? CUBLAS_OP_T : CUBLAS_OP_N, 
													static_cast<int>(Matrix<NumericType>::rows()),
													static_cast<int>(Matrix<NumericType>::columns()),
													static_cast<int>(matrixQR.columns()), // Maybe incorrect!?
													matrixQR.get(),
													static_cast<int>(matrixQR.leadingDimension()),
													tau.get(),
													Matrix<NumericType>::get(),
													static_cast<int>(Matrix<NumericType>::leadingDimension()),
													workspace.get(),
													static_cast<int>(workspace.elements()),
													devInfo.get()));
		}

		static void solveLinearEquationSystem(SideType sideOfX, FillType leftSideFill, const DeviceMatrix<NumericType>& leftSide, bool transposeLeftSide, DeviceMatrix<NumericType>& rightSide) {
			NumericType alpha = 1.0;
			CUBLAS_CALL(cublasSetStream(g_context->cublasHandle, nullptr));
			CUBLAS_CALL(wrapper::cublasXtrsm(sideOfX == SideType::LeftSide ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
											 leftSideFill == FillType::Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
											 transposeLeftSide ? CUBLAS_OP_T : CUBLAS_OP_N,
											 CUBLAS_DIAG_NON_UNIT,
											 static_cast<int>(rightSide.rows()),
											 static_cast<int>(rightSide.columns()),
											 &alpha,
											 leftSide.get(),
											 static_cast<int>(leftSide.leadingDimension()),
											 rightSide.get(),
											 static_cast<int>(rightSide.leadingDimension())));
		}

		void setNegativeToZero(cudaStream_t stream = nullptr) {
			kernel::makeNonNegative(description(), stream);
		}
	};

	template<typename NumericType>
	details::DeferredMultiplication<NumericType> operator * (const DeviceMatrix<NumericType>& left, const DeviceMatrix<NumericType>& right) {
		return details::DeferredMultiplication<NumericType>(left, false, right, false);
	}

	template<typename NumericType>
	details::DeferredMultiplication<NumericType> operator * (details::TransposedDeviceMatrix<NumericType> left, const DeviceMatrix<NumericType>& right) {
		return details::DeferredMultiplication<NumericType>(std::move(left.matrix), true, right, false);
	}

	template<typename NumericType>
	details::DeferredMultiplication<NumericType> operator * (const DeviceMatrix<NumericType>& left, details::TransposedDeviceMatrix<NumericType> right) {
		return details::DeferredMultiplication<NumericType>(left, false, std::move(right.matrix), true);
	}

	namespace details {
		template<typename NumericType>
		details::DeferredMultiplication<NumericType> operator * (details::TransposedDeviceMatrix<NumericType> left, details::TransposedDeviceMatrix<NumericType> right) {
			return details::DeferredMultiplication<NumericType>(std::move(left.matrix), true, std::move(right.matrix), true);
		}
	}
}