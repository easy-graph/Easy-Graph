import unittest

import easygraph as eg


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_overall(self):
        eg.draw_SHS_center(self.G, [1, 33, 34], style="side")
        eg.draw_SHS_center(self.G, [1, 33, 34], style="center")
        eg.draw_SHS_center_kk(self.G, [1, 33, 34], style="side")
        eg.draw_SHS_center_kk(self.G, [1, 33, 34], style="center")
        eg.draw_kamada_kawai(self.G, style="side")
        eg.draw_kamada_kawai(self.G, style="center")
        eg.draw_SHS_center(self.G, [1, 33, 34], rate=0.8, style="side")
        eg.draw_SHS_center(self.G, [1, 33, 34], rate=0.8, style="center")
        eg.draw_SHS_center_kk(self.G, [1, 33, 34], rate=0.8, style="side")
        eg.draw_SHS_center_kk(self.G, [1, 33, 34], rate=0.8, style="center")
        eg.draw_kamada_kawai(self.G, rate=0.8, style="side")
        eg.draw_kamada_kawai(self.G, rate=0.8, style="center")


if __name__ == "__main__":
    unittest.main()
    # pretty awesome images
