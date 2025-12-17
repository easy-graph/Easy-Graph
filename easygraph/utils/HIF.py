import json
import requests
import fastjsonschema
from copy import deepcopy
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from easygraph.classes.hypergraph import Hypergraph

schema_url = "https://raw.githubusercontent.com/pszufe/HIF_validators/main/schemas/hif_schema_v0.1.0.json"

class EasyGraphHIFError(Exception):
    """Custom exception for HIF conversion errors."""
    pass

_hif_validator = None

def _get_hif_validator():
    global _hif_validator
    if _hif_validator is None:
        try:
            resp = requests.get(schema_url, timeout=5)
            if resp.status_code == 200:
                schema = json.loads(resp.text)
                _hif_validator = fastjsonschema.compile(schema)
        except Exception:
            print("Warning: HIF Schema could not be fetched. Validation skipped.")
            _hif_validator = lambda x: True
            
    return _hif_validator if _hif_validator else (lambda x: True)

def hypergraph_to_hif(
    hg: Hypergraph,
    filename: Optional[Union[str, Path]] = None,
    node_label: str = "name",
    edge_label: str = "name",
) -> dict:
    """
    Converts an EasyGraph Hypergraph to HIF JSON.
    Correctly handles hg.e tuple structure ((edges), (weights), (props)).
    """
    
    if hasattr(hg, "custom_hif_nodes"):
        nodj = hg.custom_hif_nodes
    else:
        nodj = []
        num_v = hg.num_v if hasattr(hg, "num_v") else len(hg.v_property) if hasattr(hg, "v_property") else 0
        v_props = getattr(hg, "v_property", [{} for _ in range(num_v)])
        if not v_props and num_v > 0: v_props = [{} for _ in range(num_v)]

        for i in range(num_v):
            props = v_props[i] if i < len(v_props) and isinstance(v_props[i], dict) else {}
            p = props.copy()
            weight = p.pop("weight", 1.0)
            if node_label in p:
                node_id = str(p.get(node_label))
                if node_label == "name":
                    p.pop("name", None)
            else:
                node_id = p.pop("name", str(i))
            nodj.append({"node": node_id, "weight": weight, "attrs": p})

    e_structure = []
    e_weights = []
    e_props = []

    if hasattr(hg, "e") and isinstance(hg.e, tuple) and len(hg.e) == 3 and \
       isinstance(hg.e[0], (list, tuple)) and isinstance(hg.e[1], (list, tuple)):
        e_structure = hg.e[0]
        e_weights = hg.e[1]
        e_props = hg.e[2]
        
    elif hasattr(hg, "e_list") and hg.e_list:
        e_structure = hg.e_list
        e_weights = getattr(hg, "e_weight", [1.0] * len(e_structure))
        e_props = getattr(hg, "e_property_full", [{} for _ in range(len(e_structure))])
        
    elif hasattr(hg, "e") and isinstance(hg.e, (list, tuple)):
        e_structure = hg.e
        e_weights = getattr(hg, "e_weight", [1.0] * len(e_structure))
        e_props = getattr(hg, "e_property_full", [{} for _ in range(len(e_structure))])

    num_e = len(e_structure)
    
    if len(e_weights) < num_e: e_weights = [1.0] * num_e
    if len(e_props) < num_e: e_props = [{} for _ in range(num_e)]

    if hasattr(hg, "custom_hif_edges"):
        edgj = hg.custom_hif_edges
    else:
        edgj = []
        for i in range(num_e):
            props = e_props[i].copy() if isinstance(e_props[i], dict) else {}
            # edge_id = props.pop("name", str(i))
            weight = e_weights[i]
            props.pop("weight", None)
            if edge_label in props:
                edge_id = str(props.get(edge_label))
                if edge_label == "name":
                    props.pop("name", None)
            else:
                edge_id = props.pop("name", str(i))    
            edgj.append({"edge": edge_id, "weight": weight, "attrs": props})

    if hasattr(hg, "custom_hif_incidences"):
        incj = hg.custom_hif_incidences
    else:
        incj = []
        node_id_list = [n["node"] for n in nodj]
        edge_id_list = [e["edge"] for e in edgj]
        
        for e_idx, nodes_in_edge in enumerate(e_structure):
            if e_idx >= len(edge_id_list): break
            edge_name = edge_id_list[e_idx]
            
            flat_nodes = []
            if isinstance(nodes_in_edge, (list, tuple)):
                for item in nodes_in_edge:
                    if isinstance(item, (list, tuple)):
                        flat_nodes.extend(item)
                    else:
                        flat_nodes.append(item)
            else:
                flat_nodes = [nodes_in_edge]

            for n_idx in flat_nodes:
                try:
                    n_idx_int = int(n_idx)
                    if 0 <= n_idx_int < len(node_id_list):
                        incj.append({
                            "edge": edge_name,
                            "node": node_id_list[n_idx_int],
                            "weight": 1.0, 
                        })
                except (ValueError, TypeError):
                    continue

    metadata = getattr(hg, "metadata", {})
    network_type = getattr(hg, "network_type", "undirected")

    hif = {
        "nodes": nodj,
        "edges": edgj,
        "incidences": incj,
        "network-type": network_type,
        "metadata": metadata
    }

    try:
        validator = _get_hif_validator()
        validator(hif)
    except Exception as e:
        print(f"Validation Warning: {e}") 

    if filename:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(hif, f, indent=4, ensure_ascii=False)
            
    return hif


def hif_to_hypergraph(
    hif: dict = None,
    filename: Optional[Union[str, Path]] = None,
    node_label: str = "name",
    edge_label: str = "name",
):
    """
    Reads HIF JSON and returns an EasyGraph Hypergraph.
    Attaches original JSON parts to 'custom_hif_*' attributes to preserve
    structure during round-trips.
    """
    if hif is None:
        if filename is None:
            raise EasyGraphHIFError("No HIF data or filename provided.")
        try:
            with open(filename, "r", encoding='utf-8') as f:
                hif = json.load(f)
        except Exception as e:
            raise EasyGraphHIFError(f"Failed to load HIF file {filename}: {e}")

    nodes_list = hif.get("nodes", [])
    node_name_to_idx = {rec["node"]: i for i, rec in enumerate(nodes_list)}
    num_v = len(nodes_list)

    edges_list = hif.get("edges", [])
    edge_name_to_idx = {rec["edge"]: i for i, rec in enumerate(edges_list)}
    num_e = len(edges_list)

    v_property = [{} for _ in range(num_v)]
    for rec in nodes_list:
        idx = node_name_to_idx.get(rec["node"])
        if idx is not None:

            prop = rec.get("attrs", {}).copy()
            if node_label in prop:
                prop["name"] = str(prop[node_label])
            else:
                prop["name"] = rec["node"]
            prop["weight"] = rec.get("weight", 1.0)
            v_property[idx] = prop

    e_property_full = [{} for _ in range(num_e)]
    e_weight = [1.0] * num_e
    
    for rec in edges_list:
        idx = edge_name_to_idx.get(rec["edge"])
        if idx is not None:
            prop = rec.get("attrs", {}).copy()
            # if "name" not in prop:
            #     prop["name"] = rec["edge"]
            if edge_label in prop:
                prop["name"] = str(prop[edge_label])
            else:
                prop["name"] = rec["edge"]
            prop["weight"] = rec.get("weight", 1.0)
            e_property_full[idx] = prop
            e_weight[idx] = prop["weight"]

    raw_groups = [[] for _ in range(num_e)]
    
    incidences_list = hif.get("incidences", [])
    
    for inc in incidences_list:
        e_name = inc.get("edge")
        n_name = inc.get("node")
        
        e_idx = edge_name_to_idx.get(e_name)
        n_idx = node_name_to_idx.get(n_name)
        
        if e_idx is not None and n_idx is not None:
            raw_groups[e_idx].append(n_idx)

    hg = Hypergraph(
        num_v=num_v,
        e_list=raw_groups,
        e_weight=e_weight,
        v_property=v_property
    )
    
    hg.node_label_index = {}
    for i in range(num_v):
        name = v_property[i].get("name")
        if name:
            hg.node_label_index[name] = i
            
    hg.edge_label_index = {}
    for i in range(num_e):
        name = e_property_full[i].get("name")
        if name:
            hg.edge_label_index[name] = i
            
    hg.custom_hif_nodes = deepcopy(nodes_list)
    hg.custom_hif_edges = deepcopy(edges_list)
    hg.custom_hif_incidences = deepcopy(incidences_list)
    
    if "metadata" in hif:
        hg.metadata = deepcopy(hif["metadata"])
    else:
        hg.metadata = {}
        
    if "network-type" in hif:
        hg.network_type = hif["network-type"]
        
    hg.e_property_full = e_property_full

    return hg