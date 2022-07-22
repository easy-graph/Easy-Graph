import io
import sys
import time

import easygraph as eg
import pytest


class TestGEXF:
    @classmethod
    def setup_class(cls):
        cls.simple_directed_data = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
    <graph mode="static" defaultedgetype="directed">
        <nodes>
            <node id="0" label="Hello" />
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1" />
        </edges>
    </graph>
</gexf>
"""
        cls.simple_directed_graph = eg.DiGraph()
        cls.simple_directed_graph.add_node("0", label="Hello")
        cls.simple_directed_graph.add_node("1", label="World")
        cls.simple_directed_graph.add_edge("0", "1", id="0")

        cls.simple_directed_fh = io.BytesIO(cls.simple_directed_data.encode("UTF-8"))

        cls.attribute_data = """<?xml version="1.0" encoding="UTF-8"?>\
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.\
org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/\
1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
  <meta lastmodifieddate="2009-03-20">
    <creator>Gephi.org</creator>
    <description>A Web network</description>
  </meta>
  <graph defaultedgetype="directed">
    <attributes class="node">
      <attribute id="0" title="url" type="string"/>
      <attribute id="1" title="indegree" type="integer"/>
      <attribute id="2" title="frog" type="boolean">
        <default>true</default>
      </attribute>
    </attributes>
    <nodes>
      <node id="0" label="Gephi">
        <attvalues>
          <attvalue for="0" value="https://gephi.org"/>
          <attvalue for="1" value="1"/>
          <attvalue for="2" value="false"/>
        </attvalues>
      </node>
      <node id="1" label="Webatlas">
        <attvalues>
          <attvalue for="0" value="http://webatlas.fr"/>
          <attvalue for="1" value="2"/>
          <attvalue for="2" value="false"/>
        </attvalues>
      </node>
      <node id="2" label="RTGI">
        <attvalues>
          <attvalue for="0" value="http://rtgi.fr"/>
          <attvalue for="1" value="1"/>
          <attvalue for="2" value="true"/>
        </attvalues>
      </node>
      <node id="3" label="BarabasiLab">
        <attvalues>
          <attvalue for="0" value="http://barabasilab.com"/>
          <attvalue for="1" value="1"/>
          <attvalue for="2" value="true"/>
        </attvalues>
      </node>
    </nodes>
    <edges>
      <edge id="0" source="0" target="1" label="foo"/>
      <edge id="1" source="0" target="2"/>
      <edge id="2" source="1" target="0"/>
      <edge id="3" source="2" target="1"/>
      <edge id="4" source="0" target="3"/>
    </edges>
  </graph>
</gexf>
"""
        cls.attribute_graph = eg.DiGraph()
        cls.attribute_graph.graph["node_default"] = {"frog": True}
        cls.attribute_graph.add_node(
            "0", label="Gephi", url="https://gephi.org", indegree=1, frog=False
        )
        cls.attribute_graph.add_node(
            "1", label="Webatlas", url="http://webatlas.fr", indegree=2, frog=False
        )
        cls.attribute_graph.add_node(
            "2", label="RTGI", url="http://rtgi.fr", indegree=1, frog=True
        )
        cls.attribute_graph.add_node(
            "3",
            label="BarabasiLab",
            url="http://barabasilab.com",
            indegree=1,
            frog=True,
        )
        cls.attribute_graph.add_edge("0", "1", id="0", label="foo")
        cls.attribute_graph.add_edge("0", "2", id="1")
        cls.attribute_graph.add_edge("1", "0", id="2")
        cls.attribute_graph.add_edge("2", "1", id="3")
        cls.attribute_graph.add_edge("0", "3", id="4")
        cls.attribute_fh = io.BytesIO(cls.attribute_data.encode("UTF-8"))

        cls.simple_undirected_data = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
    <graph mode="static" defaultedgetype="undirected">
        <nodes>
            <node id="0" label="Hello" />
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1" />
        </edges>
    </graph>
</gexf>
"""
        cls.simple_undirected_graph = eg.Graph()
        cls.simple_undirected_graph.add_node("0", label="Hello")
        cls.simple_undirected_graph.add_node("1", label="World")
        cls.simple_undirected_graph.add_edge("0", "1", id="0")

        cls.simple_undirected_fh = io.BytesIO(
            cls.simple_undirected_data.encode("UTF-8")
        )

    def test_read_simple_directed_graphml(self):
        G = self.simple_directed_graph
        H = eg.read_gexf(self.simple_directed_fh)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(G.edges) == sorted(H.edges)
        self.simple_directed_fh.seek(0)

    def test_write_read_simple_directed_graphml(self):
        G = self.simple_directed_graph
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(G.edges) == sorted(H.edges)
        self.simple_directed_fh.seek(0)

    def test_read_simple_undirected_graphml(self):
        G = self.simple_undirected_graph
        H = eg.read_gexf(self.simple_undirected_fh)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(G.edges) == sorted(H.edges)
        self.simple_undirected_fh.seek(0)

    def test_read_attribute_graphml(self):
        G = self.attribute_graph
        H = eg.read_gexf(self.attribute_fh)
        assert sorted(G.nodes) == sorted(H.nodes)
        ge = sorted(G.edges)
        he = sorted(H.edges)
        for a, b in zip(ge, he):
            assert a == b
        self.attribute_fh.seek(0)

    def test_directed_edge_in_undirected(self):
        s = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version='1.2'>
    <graph mode="static" defaultedgetype="undirected" name="">
        <nodes>
            <node id="0" label="Hello" />
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1" type="directed"/>
        </edges>
    </graph>
</gexf>
"""
        fh = io.BytesIO(s.encode("UTF-8"))
        pytest.raises(eg.EasyGraphError, eg.read_gexf, fh)

    def test_undirected_edge_in_directed(self):
        s = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version='1.2'>
    <graph mode="static" defaultedgetype="directed" name="">
        <nodes>
            <node id="0" label="Hello" />
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1" type="undirected"/>
        </edges>
    </graph>
</gexf>
"""
        fh = io.BytesIO(s.encode("UTF-8"))
        pytest.raises(eg.EasyGraphError, eg.read_gexf, fh)

    def test_key_raises(self):
        s = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version='1.2'>
    <graph mode="static" defaultedgetype="directed" name="">
        <nodes>
            <node id="0" label="Hello">
              <attvalues>
                <attvalue for='0' value='1'/>
              </attvalues>
            </node>
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1" type="undirected"/>
        </edges>
    </graph>
</gexf>
"""
        fh = io.BytesIO(s.encode("UTF-8"))
        pytest.raises(eg.EasyGraphError, eg.read_gexf, fh)

    def test_relabel(self):
        s = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version='1.2'>
    <graph mode="static" defaultedgetype="directed" name="">
        <nodes>
            <node id="0" label="Hello" />
            <node id="1" label="Word" />
        </nodes>
        <edges>
            <edge id="0" source="0" target="1"/>
        </edges>
    </graph>
</gexf>
"""
        fh = io.BytesIO(s.encode("UTF-8"))
        G = eg.read_gexf(fh, relabel=True)
        assert sorted(G.nodes) == ["Hello", "Word"]

    def test_default_attribute(self):
        G = eg.Graph()
        G.add_node(1, label="1", color="green")
        eg.add_path(G, [0, 1, 2, 3])
        G.add_edge(1, 2, foo=3)
        G.graph["node_default"] = {"color": "yellow"}
        G.graph["edge_default"] = {"foo": 7}
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert sorted(G.nodes) == sorted(H.nodes)
        # Reading a gexf graph always sets mode attribute to either
        # 'static' or 'dynamic'. Remove the mode attribute from the
        # read graph for the sake of comparing remaining attributes.
        del H.graph["mode"]
        assert G.graph == H.graph

    def test_serialize_ints_to_strings(self):
        G = eg.Graph()
        G.add_node(1, id=7, label=77)
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert list(H) == [7]
        assert H.nodes[7]["label"] == "77"

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="requires >= python3.8")
    def test_edge_id_construct(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1, {"id": 0}), (1, 2, {"id": 2}), (2, 3)])

        expected = f"""<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi\
="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.\
gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">
    <creator>EasyGraph</creator>
  </meta>
  <graph defaultedgetype="undirected" mode="static" name="">
    <nodes>
      <node id="0" label="0" />
      <node id="1" label="1" />
      <node id="2" label="2" />
      <node id="3" label="3" />
    </nodes>
    <edges>
      <edge source="0" target="1" id="0" />
      <edge source="1" target="2" id="2" />
      <edge source="2" target="3" id="1" />
    </edges>
  </graph>
</gexf>"""

        obtained = "\n".join(eg.generate_gexf(G))
        assert expected == obtained

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="requires >= python3.8")
    def test_numpy_type(self):
        np = pytest.importorskip("numpy")
        G = eg.path_graph(4)
        eg.set_node_attributes(G, {n: n for n in np.arange(4)}, "number")
        G[0][1]["edge-number"] = np.float64(1.1)
        expected = f"""<gexf xmlns="http://www.gexf.net/1.2draft"\
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation\
="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd"\
 version="1.2">
  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">
    <creator>EasyGraph</creator>
  </meta>
  <graph defaultedgetype="undirected" mode="static" name="">
    <attributes mode="static" class="edge">
      <attribute id="1" title="edge-number" type="float" />
    </attributes>
    <attributes mode="static" class="node">
      <attribute id="0" title="number" type="int" />
    </attributes>
    <nodes>
      <node id="0" label="0">
        <attvalues>
          <attvalue for="0" value="0" />
        </attvalues>
      </node>
      <node id="1" label="1">
        <attvalues>
          <attvalue for="0" value="1" />
        </attvalues>
      </node>
      <node id="2" label="2">
        <attvalues>
          <attvalue for="0" value="2" />
        </attvalues>
      </node>
      <node id="3" label="3">
        <attvalues>
          <attvalue for="0" value="3" />
        </attvalues>
      </node>
    </nodes>
    <edges>
      <edge source="0" target="1" id="0">
        <attvalues>
          <attvalue for="1" value="1.1" />
        </attvalues>
      </edge>
      <edge source="1" target="2" id="1" />
      <edge source="2" target="3" id="2" />
    </edges>
  </graph>
</gexf>"""
        obtained = "\n".join(eg.generate_gexf(G))
        assert expected == obtained

    def test_bool(self):
        G = eg.Graph()
        G.add_node(1, testattr=True)
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert H.nodes[1]["testattr"]

    def test_specials(self):
        from math import isnan

        inf, nan = float("inf"), float("nan")
        G = eg.Graph()
        G.add_node(1, testattr=inf, strdata="inf", key="a")
        G.add_node(2, testattr=nan, strdata="nan", key="b")
        G.add_node(3, testattr=-inf, strdata="-inf", key="c")

        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        filetext = fh.read()
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)

        assert b"INF" in filetext
        assert b"NaN" in filetext
        assert b"-INF" in filetext

        assert H.nodes[1]["testattr"] == inf
        assert isnan(H.nodes[2]["testattr"])
        assert H.nodes[3]["testattr"] == -inf

        assert H.nodes[1]["strdata"] == "inf"
        assert H.nodes[2]["strdata"] == "nan"
        assert H.nodes[3]["strdata"] == "-inf"

        assert H.nodes[1]["easygraph_key"] == "a"
        assert H.nodes[2]["easygraph_key"] == "b"
        assert H.nodes[3]["easygraph_key"] == "c"

    def test_simple_list(self):
        G = eg.Graph()
        list_value = [(1, 2, 3), (9, 1, 2)]
        G.add_node(1, key=list_value)
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert H.nodes[1]["easygraph_key"] == list_value

    def test_dynamic_mode(self):
        G = eg.Graph()
        G.add_node(1, label="1", color="green")
        G.graph["mode"] = "dynamic"
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(sorted(e) for e in G.edges) == sorted(sorted(e) for e in H.edges)

    def test_slice_and_spell(self):
        # Test spell first, so version = 1.2
        G = eg.Graph()
        G.add_node(0, label="1", color="green")
        G.nodes[0]["spells"] = [(1, 2)]
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(sorted(e) for e in G.edges) == sorted(sorted(e) for e in H.edges)

        G = eg.Graph()
        G.add_node(0, label="1", color="green")
        G.nodes[0]["slices"] = [(1, 2)]
        fh = io.BytesIO()
        eg.write_gexf(G, fh, version="1.1draft")
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(sorted(e) for e in G.edges) == sorted(sorted(e) for e in H.edges)

    def test_add_parent(self):
        G = eg.Graph()
        G.add_node(0, label="1", color="green", parents=[1, 2])
        fh = io.BytesIO()
        eg.write_gexf(G, fh)
        fh.seek(0)
        H = eg.read_gexf(fh, node_type=int)
        assert sorted(G.nodes) == sorted(H.nodes)
        assert sorted(sorted(e) for e in G.edges) == sorted(sorted(e) for e in H.edges)
