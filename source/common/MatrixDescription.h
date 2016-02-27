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