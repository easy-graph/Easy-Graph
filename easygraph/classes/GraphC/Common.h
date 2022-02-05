#pragma once
#include "STLHead.h"

extern PyTypeObject GraphMapIterType;
extern PyMappingMethods GraphMapIter_mapping_methods;

extern PyMethodDef GraphMapMethods[];
extern PyMappingMethods GraphMap_mapping_methods;
extern PySequenceMethods GraphMap_sequence_methods;
extern PyTypeObject GraphMapType;

extern PySequenceMethods GraphEdge_sequence_methods;
extern PyTypeObject GraphEdgeType;

extern PyTypeObject GraphEdgesIterType;

extern PySequenceMethods GraphEdges_sequence_methods;
extern PyTypeObject GraphEdgesType;

extern PyGetSetDef Graph_get_set[];
extern PyMethodDef GraphMethods[];
extern PySequenceMethods Graph_sequence_methods;
extern PyMappingMethods Graph_mapping_methods;
extern PyTypeObject GraphType;

extern PyMethodDef ModuleMethods[];
extern PyModuleDef Graphmodule;

typedef std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>> mimimsf_t;
typedef std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>* mimimsf_pt;
typedef std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>::iterator mimimsf_it;

typedef std::unordered_map<int, std::map<std::string, float>> mimsf_t;
typedef std::unordered_map<int, std::map<std::string, float>>* mimsf_pt;
typedef std::unordered_map<int, std::map<std::string, float>>::iterator mimsf_it;

typedef std::map<std::string, float> msf_t;
typedef std::map<std::string, float>* msf_pt;
typedef std::map<std::string, float>::iterator msf_it;

typedef std::unordered_map<int, float> mif_t;
typedef std::unordered_map<int, float>* mif_pt;
typedef std::unordered_map<int, float>::iterator mif_it;

typedef std::unordered_map<int, int> mii_t;
typedef std::unordered_map<int, int>* mii_pt;
typedef std::unordered_map<int, int>::iterator mii_it;
