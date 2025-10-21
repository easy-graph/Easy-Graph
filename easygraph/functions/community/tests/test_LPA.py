import unittest

import easygraph as eg


class LPATest(unittest.TestCase):
    def test_LPA(self):
        test_graph = eg.DiGraph([(4, 6), (6, 8), (8, 10), (12, 14)])
        lpa_expected_result = [4, 6, 8, 10, 12, 14]
        lpa_actual_result = self._get_sorted_values_from_dict(eg.LPA(test_graph))
        self.assertListEqual(lpa_expected_result, lpa_actual_result)

    def test_SLPA(self):
        test_graph = eg.Graph([(1, 2), (3, 4), (4, 5)])
        slpa_expected_result = [1, 2, 3, 4, 5]
        slpa_actual_result = self._get_sorted_values_from_dict(
            eg.SLPA(test_graph, 20, 0.05)
        )
        self.assertListEqual(slpa_expected_result, slpa_actual_result)

    def test_HANP(self):
        test_graph = eg.DiGraph([(4, 6), (6, 8), (8, 10), (12, 14)])
        hanp_expected_result = [4, 6, 8, 10, 12, 14]
        hanp_actual_result = self._get_sorted_values_from_dict(
            eg.HANP(test_graph, 0.1, 0.1)
        )
        self.assertListEqual(hanp_actual_result, hanp_expected_result)

    def test_BMLPA(self):
        test_graph = eg.DiGraph([(4, 5), (6, 7), (8, 10), (10, 14)])
        bmlpa_expected_result = [4, 5, 6, 7, 8, 10, 14]
        bmlpa_actual_result = self._get_sorted_values_from_dict(
            eg.BMLPA(test_graph, 0.1)
        )
        self.assertListEqual(bmlpa_expected_result, bmlpa_actual_result)

    def _get_sorted_values_from_dict(self, res):
        return_list = []
        for val in res.values():
            for x in val:
                return_list.append(x)
        return_list.sort()
        return return_list


if __name__ == "__main__":
    unittest.main()
