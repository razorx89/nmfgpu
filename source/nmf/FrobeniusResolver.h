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

#include <common/Memory.h>

namespace nmfgpu {
	double resolveFrobenius(const HostMemory<float>& hpsVTV, const HostMemory<float>& hpsHTWTV, const HostMemory<float>& hpsHHTWTW);
	double resolveFrobenius(const HostMemory<double>& hpsVTV, const HostMemory<double>& hpsHTWTV, const HostMemory<double>& hpsHHTWTW);
}