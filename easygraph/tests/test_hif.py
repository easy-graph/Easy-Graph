import unittest
import json
import os
import tempfile
import easygraph as eg
from pathlib import Path

MOCK_HIF_DATA = {
    "metadata": {
        "name": "test_organism",
        "description": "Simulation for unit test"
    },
    "network-type": "directed",
    "nodes": [
        {"node": "n1", "weight": 1.0, "attrs": {"name": "Node A"}},
        {"node": "n2", "weight": 1.0, "attrs": {"name": "Node B"}}
    ],
    "edges": [
        {"edge": "e1", "weight": 1.0, "attrs": {"name": "Edge Alpha"}}
    ],
    "incidences": [
        {"edge": "e1", "node": "n1", "weight": 1.0, "direction": "tail"},
        {"edge": "e1", "node": "n2", "weight": 1.0, "direction": "head"}
    ]
}

class HIFTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        self.input_file = self.temp_dir_path / "input_mock.hif.json"
        self.output_file = self.temp_dir_path / "output_result.hif.json"

        with open(self.input_file, "w", encoding="utf-8") as f:
            json.dump(MOCK_HIF_DATA, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hif_roundtrip_preservation(self):
        """
        Test that custom attributes are preserved AND the generated
        EasyGraph object is structurally valid.
        """
        hg = eg.hif_to_hypergraph(filename=self.input_file)
       
        self.assertEqual(hg.num_v, 2, "Loaded graph should have 2 nodes")
        self.assertEqual(hg.num_e, 1, "Loaded graph should have 1 edge")
        
        node_names = [props.get('name') for props in hg.v_property]
        self.assertIn("n1", node_names, "Node ID 'n1' should be in v_property")
        
        edges = hg.e[0]
        self.assertEqual(len(edges), 1, "Should have 1 edge group")
        self.assertEqual(len(edges[0]), 2, "Edge e1 should connect 2 nodes")
        
        self.assertTrue(hasattr(hg, "custom_hif_incidences"), "Failed to attach custom incidences")
        self.assertTrue(hasattr(hg, "metadata"), "Failed to attach metadata")

        eg.hypergraph_to_hif(hg, filename=self.output_file)
        
        with open(self.output_file, 'r', encoding="utf-8") as f:
            res = json.load(f)
            
            first_incidence = res["incidences"][0]
            self.assertIn("direction", first_incidence, "'direction' field lost in roundtrip")
            self.assertIn(first_incidence["direction"], ["tail", "head"])
            
            self.assertNotIn("default_attrs", res["metadata"], "'default_attrs' was forced into metadata")
            self.assertEqual(res["metadata"]["name"], "test_organism")

    def test_manual_graph_export(self):
        """Test exporting a manually created Hypergraph (not loaded from file)."""
        hg = eg.Hypergraph(
            num_v=5, 
            e_list=[(0, 1, 2), (2, 3), (2, 3), (0, 4)], 
            merge_op="sum"
        )
        hg.metadata = {"created_by": "manual_test"}

        eg.hypergraph_to_hif(hg, filename=self.output_file)
        
        with open(self.output_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data["nodes"]), 5)
            self.assertEqual(len(data["edges"]), 3) 
            self.assertEqual(data["metadata"]["created_by"], "manual_test")
            

            weights = [e["weight"] for e in data["edges"]]
            self.assertIn(2.0, weights, "Merged edge weight should be 2.0")

if __name__ == "__main__":
    unittest.main()