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

namespace nmfgpu {
	//enum class StorageFormat {
	//	/** Dense Matrix */
	//	Dense,
	//	/** Compressed Sparse Row */
	//	CSR,
	//	/** Compressed Sparse Column */
	//	CSC,
	//	/** Coordinate */
	//	COO
	//};

	//template<typename NumericType>
	//struct MatrixDescription {
	//	/** Number of rows of the matrix */
	//	unsigned rows;

	//	/** Number of columns of the matrix. */
	//	unsigned columns;

	//	/** Storage format of the matrix. */
	//	StorageFormat format;
	//	union {
	//		/** Description of a dense matrix if the storage format is equal to StorageFormat::Dense. */
	//		struct {
	//			NumericType* values;
	//			unsigned leadingDimension;
	//		} dense;

	//		/** Description of a compressed sparse row matrix if the storage format is equal to StorageFormat::CSR. */
	//		struct {
	//			NumericType* values;
	//			int* rowPtr;
	//			int* columnIndices;
	//			unsigned nnz;
	//			IndexBase base;
	//		} csr;

	//		/** Description of a compressed sparse column matrix if the storage format is equal to StorageFormat::CSC. */
	//		struct {
	//			NumericType* values;
	//			int* columnPtr;
	//			int* rowIndices;
	//			unsigned nnz;
	//			IndexBase base;
	//		} csc;

	//		/** Description of a sparse matrix stored in coordinate format, equal to an uncompressed StorageFormat::CSR format. */
	//		struct {
	//			NumericType* values;
	//			int* rowIndices;
	//			int* columnIndices;
	//			unsigned nnz;
	//			IndexBase base;
	//		} coo;
	//	};
	//};
}