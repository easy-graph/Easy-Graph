#include "operation.h"
#include "graph.h"
double density(const Graph& G) {
	const std::size_t n = G.node.size();
	if (n <= 1) return 0.0;

	// m2 为度数总和；无向图中 m2 = 2 * |E|
	std::size_t m2 = 0;
	for (const auto& kv : G.adj) {
		m2 += kv.second.size();
	}
	if (m2 == 0) return 0.0;

	// 与原实现一致：d = m2 / (n*(n-1))
	// 对无向简单图，这等价于 2|E| / (n*(n-1))
	return static_cast<double>(m2) /
		   (static_cast<double>(n) * static_cast<double>(n - 1));
}
