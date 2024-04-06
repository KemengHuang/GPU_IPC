#pragma once
#include <string_view>

namespace gipc
{
	constexpr auto assets_dir()
	{
		return std::string_view{ GIPC_ASSETS_DIR };
	}

	constexpr auto output_dir()
	{
		return std::string_view{ GIPC_OUTPUT_DIR };
	}
}
