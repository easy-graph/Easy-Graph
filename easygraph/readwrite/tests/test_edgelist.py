import unittest

import easygraph as eg


class ParseEdgeList(unittest.TestCase):

    def test_parse_edgelist(self):
        self.assertEqual(self.run_parse_edgelist_with_no_data(), [1, 2, 3, 4])
        self.assertEqual(self.run_parse_edgelist_with_dict(), [1, 2, 3, 4])
        self.assertEqual(self.run_parse_edgelist_with_list_data(), [1, 2, 3, 4])

    def run_parse_edgelist_with_no_data(self):
        lines = ["1 2", "2 3", "3 4"]
        return list(eg.parse_edgelist(lines, nodetype=int))

    def run_parse_edgelist_with_dict(self):
        lines = ["1 2 {'weight': 3}", "2 3 {'weight': 27}", "3 4 {'weight': 3.0}"]
        return list(eg.parse_edgelist(lines, nodetype=int))

    def run_parse_edgelist_with_list_data(self):
        lines = ["1 2 3", "2 3 27", "3 4 3.0"]
        return list(eg.parse_edgelist(lines, nodetype=int, data=(("weight", float),)))


if __name__ == '__main__':
    unittest.main()
