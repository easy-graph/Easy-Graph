#include "utils.h"

py::object attr_to_dict(const node_attr_dict_factory& attr) {
	py::dict attr_dict = py::dict();
	for (const auto& kv : attr) {
		attr_dict[py::cast(kv.first)] = kv.second;
	}
	return attr_dict;
}

std::string weight_to_string(py::object weight) {
	py::object warn = py::module_::import("warnings").attr("warn");
	if (!py::isinstance<py::str>(weight)) {
		if (!weight.is_none()) {
			warn(py::str(weight) + py::str(" would be transformed into an instance of str."));
		}
		weight = py::str(weight);
	}
	std::string weight_key = weight.cast<std::string>();
	return weight_key;
}

py::object py_sum(py::object o) {
	py::object sum = py::module_::import("builtins").attr("sum");
	return sum(o);
}