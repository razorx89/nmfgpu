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