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