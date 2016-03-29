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

#include <type_traits>

namespace nmfgpu {
	namespace traits {
		namespace detail {
			template <typename T, typename NameGetter>
			struct HasMemberImpl {
				template <typename C>
				static char f(typename NameGetter::template get<C>*);

				template <typename C>
				static long f(...);

			public:
				static const bool value = (sizeof(f<T>(0)) == sizeof(char));
			};
		}

		template <typename T, typename NameGetter>
		struct HasMember :
			std::integral_constant<bool, detail::HasMemberImpl<T, NameGetter>::value> { };
	}
}