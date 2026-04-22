"""
Shared Knowledge Graph Loader for CWQ Subgraph Traversal
"""
import json, os, sys
from collections import defaultdict

class KnowledgeGraph:
    """A simple KG subgraph built from CWQ SPARQL triples."""
    
    def __init__(self):
        # entity -> [(relation, target_entity)]
        self.forward = defaultdict(list)   # (subj) -[rel]-> (obj)
        self.backward = defaultdict(list)  # (obj) -[rel^-1]-> (subj)
        self.entities = set()
        self.relations = set()
    
    def add_triple(self, subj, rel, obj):
        """Add a triple: subj -[rel]-> obj"""
        self.forward[subj].append((rel, obj))
        self.backward[obj].append((rel, subj))
        self.entities.add(subj)
        self.entities.add(obj)
        self.relations.add(rel)
    
    def get_neighbors(self, entity):
        """Get all (relation, target) pairs reachable from entity."""
        neighbors = []
        for rel, tgt in self.forward.get(entity, []):
            neighbors.append((rel, +1, tgt))  # forward
        for rel, tgt in self.backward.get(entity, []):
            neighbors.append((rel, -1, tgt))  # backward
        return neighbors
    
    def traverse(self, start_entity, relation_path):
        """
        Traverse KG from start_entity following relation_path.
        relation_path: list of (relation_name, direction)
        Returns set of reached entities.
        """
        current = {start_entity}
        
        for rel, direction in relation_path:
            next_entities = set()
            for entity in current:
                if direction == +1:
                    for r, tgt in self.forward.get(entity, []):
                        if r == rel:
                            next_entities.add(tgt)
                else:  # backward
                    for r, tgt in self.backward.get(entity, []):
                        if r == rel:
                            next_entities.add(tgt)
            current = next_entities
            if not current:
                break
        
        return current

def build_kg_from_cwq_triples(json_paths, extractor_fn):
    """
    Build KG subgraph from all CWQ SPARQL queries.
    extractor_fn: function that takes SPARQL and returns list of (subj, rel, obj)
    """
    kg = KnowledgeGraph()
    
    for path in json_paths:
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                triples = extractor_fn(item['sparql'])
                for subj, rel, obj in triples:
                    subj_clean = subj.replace('ns:', '')
                    obj_clean = obj.replace('ns:', '')
                    if not subj.startswith('?') and not obj.startswith('?'):
                        kg.add_triple(subj_clean, rel, obj_clean)
                    elif not subj.startswith('?'):
                        kg.add_triple(subj_clean, rel, obj)
                    elif not obj.startswith('?'):
                        kg.add_triple(subj, rel, obj_clean)
                    else:
                        kg.add_triple(subj, rel, obj)
                        
    print(f"KG Stats: {len(kg.entities)} entities, {len(kg.relations)} relations, "
          f"{sum(len(v) for v in kg.forward.values())} forward edges")
    return kg
