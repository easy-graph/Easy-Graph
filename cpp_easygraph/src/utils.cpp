#include "utils.h"

py::dict attr_to_dict(const node_attr_dict_factory& attr) {
	py::dict attr_dict;
	for (const auto& kv : attr) {
		// kv.first 是 std::string，kv.second 是 weight_t
		attr_dict[py::str(kv.first)] = py::cast(kv.second);
	}
	return attr_dict;
}

std::string weight_to_string(py::handle weight) {
	// 如果 weight 不是 str，则尝试转换为 str，并在 None 情况下发警告
	if (!py::isinstance<py::str>(weight)) {
		if (weight.is_none()) {
			py::module_::import("warnings")
				.attr("warn")("None will be transformed into an instance of str.");
		}
		// 强制转为 Python 字符串
		weight = py::str(weight);
	}

	// 转成 std::string
	return py::cast<std::string>(weight);
}

py::object py_sum(const py::object& o) {
	py::object sum_func = py::module_::import("builtins").attr("sum");
	return sum_func(o);
}
