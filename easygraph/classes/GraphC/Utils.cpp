#include "Utils.h"

py::object attr_to_dict(const Graph::node_attr_dict_factory& attr) {
	py::dict attr_dict = py::dict();
	for (const auto& kv : attr) {
		attr_dict[kv.first] = kv.second;
	}
	return attr_dict;
}

std::string weight_to_string(py::object weight) {
	if (weight.attr("__class__") != py::str().attr("__class__")) {
		warn(py::str(weight) + py::str(" would be transformed into an instance of str."));
		weight = py::str(weight);
	}
	std::string weight_key = py::extract<std::string>(weight);
	return weight_key;
}