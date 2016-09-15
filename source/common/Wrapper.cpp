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

#include <common/Interface.h>
#include <common/Wrapper.h>

namespace nmfgpu {
	namespace wrapper {	
		// cuSPARSE Overloaded Wrapper Functions ----------------------------------------------------------------------------------
		
		cusparseStatus_t cusparseXnnz(cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, 
									  const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr) {
			return cusparseSnnz(g_context->cusparseHandle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
		}
	
		cusparseStatus_t cusparseXnnz(cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, 
									  const double *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr) {
			return cusparseDnnz(g_context->cusparseHandle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
		}
		
		cusparseStatus_t cusparseXdense2csr(int m, int n, const cusparseMatDescr_t descrA, const float *A, 
											int lda, const int *nnzPerRow, float *csrValA, int *csrRowPtrA, int *csrColIndA) {
			return cusparseSdense2csr(g_context->cusparseHandle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
		}
											
		cusparseStatus_t cusparseXdense2csr(int m, int n, const cusparseMatDescr_t descrA, const double *A, 
											int lda, const int *nnzPerRow, double *csrValA, int *csrRowPtrA, int *csrColIndA) {
			return cusparseDdense2csr(g_context->cusparseHandle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
		}
		
		cusparseStatus_t cusparseXdense2csc(int m, int n, const cusparseMatDescr_t descrA, const float *A, 
											int lda, const int *nnzPerCol, float *cscValA, int *cscRowPtrA, int *cscColIndA) {
			return cusparseSdense2csc(g_context->cusparseHandle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowPtrA, cscColIndA);
		}
											
		cusparseStatus_t cusparseXdense2csc(int m, int n, const cusparseMatDescr_t descrA, const double *A, 
											int lda, const int *nnzPerCol, double *cscValA, int *cscRowPtrA, int *cscColIndA) {
			return cusparseDdense2csc(g_context->cusparseHandle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowPtrA, cscColIndA);
		}		
									  
		
		cusparseStatus_t cusparseXcsc2dense(int m, int n, const cusparseMatDescr_t descrA, 
											const float *cscValA, const int *cscRowIndA, const int *cscColPtrA,
											float *A, int lda) {
			return cusparseScsc2dense(g_context->cusparseHandle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
		}

		cusparseStatus_t cusparseXcsc2dense(int m, int n, const cusparseMatDescr_t descrA, 
											const double *cscValA, const int *cscRowIndA, const int *cscColPtrA,
											double *A, int lda) {
			return cusparseDcsc2dense(g_context->cusparseHandle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda);
		}
											
		cusparseStatus_t cusparseXcsr2dense(int m, int n,const cusparseMatDescr_t descrA, const float *csrValA,
											const int *csrRowPtrA, const int *csrColIndA, float *A, int lda) {
			return cusparseScsr2dense(g_context->cusparseHandle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
		}
											
		cusparseStatus_t cusparseXcsr2dense(int m, int n,const cusparseMatDescr_t descrA, const double *csrValA,
											const int *csrRowPtrA, const int *csrColIndA, double *A, int lda) {
			return cusparseDcsr2dense(g_context->cusparseHandle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
		}

		// cuBLAS Overloaded Wrapper Functions -----------------------------------------------------------------------------------
		cublasStatus_t cublasXscal(int n, const float* alpha, float* x, int incx) {
			return cublasSscal(g_context->cublasHandle, n, alpha, x, incx);
		}

		cublasStatus_t cublasXscal(int n, const double* alpha, double* x, int incx) {
			return cublasDscal(g_context->cublasHandle, n, alpha, x, incx);
		}

		cublasStatus_t cublasXcopy(int n, const float* x, int incx, float* y, int incy) {
			return cublasScopy(g_context->cublasHandle, n, x, incx, y, incy);
		}

		cublasStatus_t cublasXcopy(int n, const double* x, int incx, double* y, int incy) {
			return cublasDcopy(g_context->cublasHandle, n, x, incx, y, incy);
		}
		
		cublasStatus_t cublasXaxpy(int n, const float* alpha, const float* x, int incx, float* y, int incy) {
			return cublasSaxpy(g_context->cublasHandle, n, alpha, x, incx, y, incy);
		}

		cublasStatus_t cublasXaxpy(int n, const double* alpha, const double* x, int incx, double* y, int incy) {
			return cublasDaxpy(g_context->cublasHandle, n, alpha, x, incx, y, incy);
		}

		cublasStatus_t cublasXdot(int n, const float *x, int incx, const float *y, int incy, float *result) {
			return cublasSdot(g_context->cublasHandle, n, x, incx, y, incy, result);
		}
	
		cublasStatus_t cublasXdot(int n, const double *x, int incx, const double *y, int incy, double *result) {
			return cublasDdot(g_context->cublasHandle, n, x, incx, y, incy, result);
		}
		
		cublasStatus_t cublasXgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, 
								   const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
			return cublasSgemm(g_context->cublasHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
								   
		cublasStatus_t cublasXgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, 
								   const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
			return cublasDgemm(g_context->cublasHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
		
		cublasStatus_t cublasXsymm(cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, 
								   const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
			return cublasSsymm(g_context->cublasHandle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
		}
								   
		cublasStatus_t cublasXsyrk(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha,
								   const float *A, int lda, const float *beta, float *C, int ldc) {
			return cublasSsyrk(g_context->cublasHandle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
		}

		cublasStatus_t cublasXsyrk(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha,
			const double *A, int lda, const double *beta, double *C, int ldc) {
			return cublasDsyrk(g_context->cublasHandle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
		}

		cublasStatus_t cublasXsymm(cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, 
								   const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
			return cublasDsymm(g_context->cublasHandle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
		}
								   
		cublasStatus_t cublasXtrsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
								   int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
			return cublasStrsm(g_context->cublasHandle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
		}
								   
		cublasStatus_t cublasXtrsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
								   int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
			return cublasDtrsm(g_context->cublasHandle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
		}
	
		cublasStatus_t cublasXgeam(cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, 
								   const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
			return cublasSgeam(g_context->cublasHandle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}
		                      	   
		cublasStatus_t cublasXgeam(cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, 
								   const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {                      	   
			return cublasDgeam(g_context->cublasHandle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}
	
		// cuSOLVER Overloaded Wrapper Functions ----------------------------------------------------------------------------------
		cusolverStatus_t cusolverDnXgeqrf_bufferSize(int m, int n, float *A, int lda, int *Lwork) {
			return cusolverDnSgeqrf_bufferSize(g_context->cusolverDnHandle, m, n, A, lda, Lwork);
		}

		cusolverStatus_t cusolverDnXgeqrf_bufferSize(int m, int n, double *A, int lda, int *Lwork) {
			return cusolverDnDgeqrf_bufferSize(g_context->cusolverDnHandle, m, n, A, lda, Lwork);
		}

		cusolverStatus_t cusolverDnXgeqrf(int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int* devInfo) {
			return cusolverDnSgeqrf(g_context->cusolverDnHandle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
		}

		cusolverStatus_t cusolverDnXgeqrf(int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int* devInfo) {
			return cusolverDnDgeqrf(g_context->cusolverDnHandle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
		}

		cusolverStatus_t cusolverDnXormqr(cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, 
			const float *A, int lda, const float *tau, float *C, int ldc, float *work, int lwork, int* devInfo) {
			return cusolverDnSormqr(g_context->cusolverDnHandle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
		}

		cusolverStatus_t cusolverDnXormqr(cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, 
			const double *A, int lda, const double *tau, double *C, int ldc, double *work, int lwork, int* devInfo) {
			return cusolverDnDormqr(g_context->cusolverDnHandle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
		}


		// MAGMA Overloaded Wrapper Functions -----------------------------------------------------------------------------------
	
		/*void magmablas_xlaset(magma_uplo_t uplo, magma_int_t m, magma_int_t n, float offdiag, float diag, magmaFloat_ptr dA, magma_int_t ldda) {
			magmablas_slaset(uplo, m, n, offdiag, diag, dA, ldda);
		}
	
		void magmablas_xlaset(magma_uplo_t uplo, magma_int_t m, magma_int_t n, double offdiag, double diag, magmaDouble_ptr dA, magma_int_t ldda) {
			magmablas_dlaset(uplo, m, n, offdiag, diag, dA, ldda);
		}
	
		void magmablas_xswapdblk(magma_int_t n, magma_int_t nb, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca, 
								 magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb) {
			magmablas_sswapdblk(n, nb, dA, ldda, inca, dB, lddb, incb);
		}
								 
		void magmablas_xswapdblk(magma_int_t n, magma_int_t nb, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t inca, 
								 magmaDouble_ptr dB, magma_int_t lddb, magma_int_t incb) {
			magmablas_dswapdblk(n, nb, dA, ldda, inca, dB, lddb, incb);
		}
			
		magma_int_t magma_xgels_gpu(magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, float* dB,
									magma_int_t lddb, float* hwork, magma_int_t lwork, magma_int_t* info) {
			return magma_sgels_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info);
		}			
	
		magma_int_t magma_xgels_gpu(magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, double* dB,
									magma_int_t lddb, double* hwork, magma_int_t lwork, magma_int_t* info) {
			return magma_dgels_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info);
		}	
		
		void magma_xtrsm(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
						 float alpha, const float* dA, magma_int_t ldda, float*	dB, magma_int_t lddb) {
			magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb);
		}
		
		void magma_xtrsm(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
						 double alpha, const double* dA, magma_int_t ldda, double* dB, magma_int_t lddb) {
			magma_dtrsm(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb);
		}
	
		magma_int_t magma_xgeqrf_gpu(magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda, float* tau, float* dT, magma_int_t* info) {
			return magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
		}
	
		magma_int_t magma_xgeqrf_gpu(magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda, double* tau, double* dT, magma_int_t* info) {
			return magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
		}
	
		magma_int_t magma_xgeqrf2_gpu(magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda, float* tau, magma_int_t* info) {
			return magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
		}
		 	
		magma_int_t magma_xgeqrf2_gpu(magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda, double* tau, magma_int_t* info) {
			return magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
		}
	
		magma_int_t magma_xgeqrf3_gpu(magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda, float* tau, float* dT, magma_int_t* info) {
			return magma_sgeqrf3_gpu(m, n, dA, ldda, tau, dT, info);
		}
	
		magma_int_t magma_xgeqrf3_gpu(magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda, double* tau, double* dT, magma_int_t* info) {
			return magma_dgeqrf3_gpu(m, n, dA, ldda, tau, dT, info);
		}
		
		magma_int_t magma_xgeqrs3_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, float* tau, float* dT, 
									  float* dB, magma_int_t lddb, float* hwork, magma_int_t lwork, magma_int_t* info) {
			return magma_sgeqrs3_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info);
		}
	
		magma_int_t magma_xgeqrs3_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, double* tau, double* dT, 
									  double* dB, magma_int_t lddb, double* hwork, magma_int_t lwork, magma_int_t* info) {
			return magma_dgeqrs3_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info);
		}
	
		magma_int_t magma_xormqr_gpu(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, 
									 float* dA,	magma_int_t ldda, float* tau, float* dC, magma_int_t lddc, float* hwork,
									 magma_int_t lwork,	float* dT, magma_int_t nb, magma_int_t*	info) {
			return magma_sormqr_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, hwork, lwork, dT, nb, info);
		}
									 	
		magma_int_t magma_xormqr_gpu(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, 
									 double* dA, magma_int_t ldda, double* tau, double* dC, magma_int_t lddc, double* hwork,
									 magma_int_t lwork,	double* dT, magma_int_t nb, magma_int_t* info) {
			return magma_dormqr_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, hwork, lwork, dT, nb, info);
		}
	
		magma_int_t magma_xormqr2_gpu(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, float* dA, magma_int_t ldda, 
									  float* tau, float* dC, magma_int_t lddc, float* wA, magma_int_t ldwa, magma_int_t* info) {
			return magma_sormqr2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info);
		}
									  	
		magma_int_t magma_xormqr2_gpu(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, double* dA, magma_int_t ldda, 
									  double* tau, double* dC, magma_int_t lddc, double* wA, magma_int_t ldwa, magma_int_t* info) {
			return magma_dormqr2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info);
		}
		
		magma_int_t magma_xgeqp3_gpu(magma_int_t m, magma_int_t n, float* A, magma_int_t lda, magma_int_t* jpvt, float* tau, float* work, magma_int_t lwork, magma_int_t* info) {
			return magma_sgeqp3_gpu(m, n, A, lda, jpvt, tau, work, lwork, info);		
		}
		
		magma_int_t magma_xgeqp3_gpu( magma_int_t m, magma_int_t n, double*	A, magma_int_t lda, magma_int_t* jpvt, double* tau, double* work, magma_int_t lwork, magma_int_t* info) {
			return magma_dgeqp3_gpu(m, n, A, lda, jpvt, tau, work, lwork, info);
		}*/
	}
}
