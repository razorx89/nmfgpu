#include <assert.h>
#include "Summary.h"

namespace nmfgpu {

	void Summary::destroy() {
		delete this;
	}

	unsigned Summary::bestRun() const {
		assert(m_records.size() > 0);
		return m_bestRun;
	}

	void Summary::record(unsigned index, ExecutionRecord& record) const {
		assert(index < m_records.size());
		if (index < m_records.size()) {
			record = m_records[index];
		}
	}

	unsigned Summary::recordCount() const {
		return unsigned(m_records.size());
	}

	void Summary::insert(const ExecutionRecord& record) {
		m_records.push_back(record);
		for (auto i = 0u; i < m_records.size() - 1; ++i) {
			if (m_records[i].frobenius <= record.frobenius) {
				return;
			}
		}
		m_bestRun = m_records.size() - 1;
	}

	void Summary::reset() {
		m_bestRun = 0;
		m_records.clear();
	}
}