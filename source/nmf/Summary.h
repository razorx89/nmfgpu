#pragma once

#include <nmfgpu.h>
#include <vector>

namespace nmfgpu {
	class Summary : public ISummary {
		std::vector<ExecutionRecord> m_records;
		unsigned m_bestRun;
	public:
		virtual void destroy() override;

		virtual unsigned bestRun() const override;

		virtual void record(unsigned index, ExecutionRecord& record) const override;

		virtual unsigned recordCount() const override;
		
		void insert(const ExecutionRecord& record);

		void reset();

	};
}