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