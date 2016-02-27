#pragma once

#include <common/MatrixDescription.h>
#include <map>
#include <nmfgpu.h>
#include <nmf/Dispatcher.h>
#include <string>

namespace nmfgpu {
#if 0
	template<typename NumericType>
	class Context : public INmfContext<NumericType> {
		NmfAlgorithm m_algorithm{ NmfAlgorithm::Multiplicative };
		MatrixDescription<NumericType> m_matrixData;
		unsigned m_features{ 0u };
		NmfInitializationMethod m_initMethod{ NmfInitializationMethod::AllRandomValues };
		unsigned m_numIterations{ 2000u };
		NumericType* m_matrixW{ nullptr };
		unsigned m_leadingDimensionW{ 0u };
		NumericType* m_matrixH{ nullptr };
		unsigned m_leadingDimensionH{ 0u };
		std::map<std::string, double> m_parameter;
		unsigned m_numRuns{ 1u };
		NmfThresholdType m_thresholdType{ NmfThresholdType::Frobenius };
		double m_thresholdValue{ 0.01 };
		UserInterruptCallback m_userInterruptCallback{ nullptr };
		unsigned m_seed;
		bool m_enableSSNMFTestDataFactorization{ false };

	public:
		virtual ~Context() = default;

		virtual void compute(ExecutionStatistic*) override {

		}

		virtual void destroy() override {
			delete this;
		}

		virtual void setAlgorithm(NmfAlgorithm algorithm) override {
			m_algorithm = algorithm;
		}

		virtual void setDataMatrixDense(unsigned rows, unsigned columns, NumericType* matrixData, unsigned leadingDimension) override {
			m_matrixData.format = StorageFormat::Dense;
			m_matrixData.rows = rows;
			m_matrixData.columns = columns;
			m_matrixData.dense.values = matrixData;
			m_matrixData.dense.leadingDimension = leadingDimension;
		}

		virtual void setDataMatrixCOO(unsigned rows, unsigned columns, unsigned nnz, NumericType* values, int* rowIndices, int* columnIndices, IndexBase base) override {
			m_matrixData.format = StorageFormat::COO;
			m_matrixData.rows = rows;
			m_matrixData.columns = columns;
			m_matrixData.coo.values = values;
			m_matrixData.coo.rowIndices = rowIndices;
			m_matrixData.coo.columnIndices = columnIndices;
			m_matrixData.coo.nnz = nnz;
			m_matrixData.coo.base = base;
		}

		virtual void setDataMatrixCSC(unsigned rows, unsigned columns, unsigned nnz, NumericType* values, int* columnPtr, int* rowIndices, IndexBase base) override {
			m_matrixData.format = StorageFormat::CSC;
			m_matrixData.rows = rows;
			m_matrixData.columns = columns;
			m_matrixData.csc.values = values;
			m_matrixData.csc.columnPtr = columnPtr;
			m_matrixData.csc.rowIndices = rowIndices;
			m_matrixData.csc.nnz = nnz;
			m_matrixData.csc.base = base;
		}

		virtual void setDataMatrixCSR(unsigned rows, unsigned columns, unsigned nnz, NumericType* values, int* rowPtr, int* columnIndices, IndexBase base) override {
			m_matrixData.format = StorageFormat::CSR;
			m_matrixData.rows = rows;
			m_matrixData.columns = columns;
			m_matrixData.csr.values = values;
			m_matrixData.csr.rowPtr = rowPtr;
			m_matrixData.csr.columnIndices = columnIndices;
			m_matrixData.csr.nnz = nnz;
			m_matrixData.csr.base = base;
		}

		virtual void setFeatureCount(unsigned features) override { m_features = features; }
		virtual void setInitializationMethod(NmfInitializationMethod method) override { m_initMethod = method; }
		virtual void setIterationCount(unsigned numIterations) { m_numIterations = numIterations; }

		virtual void setOutputMatrices(NumericType* matrixW, unsigned leadingDimensionW, NumericType* matrixH, unsigned leadingDimensionH) override {
			m_matrixW = matrixW;
			m_leadingDimensionW = leadingDimensionW;
			m_matrixH = matrixH;
			m_leadingDimensionH = leadingDimensionH;
		}

		virtual void setParameter(const char* name, double value) override { 
			m_parameter[name] = value; 
		}

		/*virtual void setParameter(const char* name, int value) override { 
			m_parameter[name] = static_cast<double>(value); 
		}*/

		virtual void setRunCount(unsigned numRuns) override { m_numRuns = numRuns; }
		virtual void setSeed(unsigned seed) override { m_seed = seed; }
		virtual void setThresholdType(NmfThresholdType type) override { m_thresholdType = type; }
		virtual void setThresholdValue(double threshold) override { m_thresholdValue = threshold; }
		virtual void setUserInterruptCallback(UserInterruptCallback callback) override { m_userInterruptCallback = callback; }

		virtual void enableSSNMFTestDataFactorization(bool enabled) override { m_enableSSNMFTestDataFactorization = enabled; }

		/** Returns the user defined algorithm which should be executed. */
		NmfAlgorithm algorithm() const { return m_algorithm; }

		/** Returns a description of the data matrix which can either be dense or sparse. */
		MatrixDescription<NumericType> dataMatrix() const { return m_matrixData; }
		
		/** Returns a description of the dense output matrix W. The description depends on the feature count and data matrix. */
		MatrixDescription<NumericType> outputMatrixW() const {
			auto desc = MatrixDescription<NumericType>();
			desc.format = StorageFormat::Dense;
			desc.rows = m_matrixData.rows;
			desc.columns = m_features;
			desc.dense.values = m_matrixW;
			desc.dense.leadingDimension = m_leadingDimensionW;
			return desc;
		}

		/** Returns a description of the dense output matrix H. The description depends on the feature count and data matrix. */
		MatrixDescription<NumericType> outputMatrixH() const {
			auto desc = MatrixDescription<NumericType>();
			desc.format = StorageFormat::Dense;
			desc.rows = m_features;
			desc.columns = m_matrixData.columns;
			desc.dense.values = m_matrixH;
			desc.dense.leadingDimension = m_leadingDimensionH;
			return desc;
		}

		/** Returns the user defined number of features. */
		unsigned features() const { return m_features; }

		/** Returns the user defined initialization method for matrix W and H. */
		NmfInitializationMethod initializationMethod() const { return m_initMethod; }

		/** Returns the user defined number of iterations. */
		unsigned iterationCount() const { return m_numIterations; }
		
		/** Checks if multiple parameters were set by the user.
		@param names List of names which must be set by the user. 
		@returns If all parameters were set then the function will return true. */
		bool hasParameters(std::initializer_list<const char*> names) const { 
			for (auto name : names) { 
				if (m_parameter.find(name) == m_parameter.end()) {
					return false;
				}
			} 
			
			return true;
		}

		/** Checks if a single parameter was set by the user. 
		@param name Name of the parameter. 
		@returns If the parameter was set by the user then the function will return true. */
		bool hasParameter(const char* name) const { 
			return m_parameter.find(name) != m_parameter.end(); 
		}
		
		/** Returns the requested user defined parameter value using a conversion into any double compatible type. 
		@tparam T Requested parameter type which can be constructed from a double value. 
		@throws std::invalid_argument Will be thrown if the requested parameter does not exist.*/
		template<typename T>
		T parameter(const char* name) const {
			auto it = m_parameter.find(name);
			if (it == m_parameter.end()) {
				throw std::invalid_argument(std::string("Parameter '") + name + "' does not exist!");
			}

			return static_cast<T>(it->second); 
		}

		/** Returns the user defined number of runs which will be performed. */
		unsigned runCount() const { return m_numRuns; }

		unsigned seed() const { return m_seed; }

		/** Returns the user defined threshold type for checking convergency. */
		NmfThresholdType thresholdType() const { return m_thresholdType; }

		/** Returns the user defined threshold value for checking convergency. */
		double thresholdValue() const { return m_thresholdValue; }

		/** Returns the user defined callback function for checking any occurred interrupt. */
		UserInterruptCallback userInterruptCallback() const { return m_userInterruptCallback; }

		DispatcherConfig dispatcherConfig() const {
			auto config = DispatcherConfig();
			config.numIterations = m_numIterations;
			config.numRuns = m_numRuns;
			config.thresholdType = m_thresholdType;
			config.thresholdValue = m_thresholdValue;
			config.userInterruptCallback = m_userInterruptCallback;
			return config;
		}

		bool isSSNMFTestDataFactorization() const { return m_enableSSNMFTestDataFactorization; }
	};
#endif
}