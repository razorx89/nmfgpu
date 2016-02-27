#pragma once

#include <memory>

namespace nmfgpu {
	/** Implementation of a C++14 make_unique function, as it is not proposed by the C++11 standard.
	@tparam T Type of the object to be constructed.
	@tparam Args List of type names for the argument list.
	@param args List of argument values which will be forwarded to the constructor of @p T.
	@returns Returns a constructed and initialized unique_ptr of type @p T
	@see http://herbsutter.com/gotw/_102/ */
	template<typename T, typename... Args>
	std::unique_ptr<T> make_unique(Args&&... args) {
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
}