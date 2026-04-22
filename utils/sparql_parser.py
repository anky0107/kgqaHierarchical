# utils/sparql_parser.py

import re
from collections import defaultdict, deque


TRIPLE_PATTERN = re.compile(
    r'(\?\w+|ns:[mg]\.[\w\d_]+)\s+ns:([\w\d\._]+)\s+(\?\w+|ns:[mg]\.[\w\d_]+)\s*\.'
)


def extract_triples(sparql):
    """
    Extract (subject, relation, object) triples from CWQ SPARQL.
    Handles standard triples and inverse relations using the ^ operator.
    Returns list of (subj, rel, obj)
    """
    triples = []
    # Standard: ns:m... ns:rel ?x .
    std_pattern = re.compile(r'(\?\w+|ns:[mg]\.[\w\d_]+)\s+ns:([\w\d\._]+)\s+(\?\w+|ns:[mg]\.[\w\d_]+)\s*\.')
    matches = std_pattern.findall(sparql)
    for subj, rel, obj in matches:
        triples.append((subj, rel, obj))
    
    # Inverse: ?y ^ns:rel ?x .  =>  equivalent to ?x ns:rel ?y .
    inv_pattern = re.compile(r'(\?\w+|ns:[mg]\.[\w\d_]+)\s+\^ns:([\w\d\._]+)\s+(\?\w+|ns:[mg]\.[\w\d_]+)\s*\.')
    inv_matches = inv_pattern.findall(sparql)
    for obj_inv, rel, subj_inv in inv_matches:
        triples.append((subj_inv, rel, obj_inv))
        
    return triples


def build_graph(triples):
    """
    Build adjacency list (bidirectional) for traversal.
    Returns:
        graph[node] = list of (neighbor, relation, direction)
    direction = +1 if forward, -1 if inverse
    """
    graph = defaultdict(list)

    for subj, rel, obj in triples:
        graph[subj].append((obj, rel, +1))
        graph[obj].append((subj, rel, -1))

    return graph


def find_entity_constants(triples):
    """
    Find entity constants (Freebase MIDs like ns:m.xxx) that appear in triples.
    These serve as starting points for BFS.
    """
    entities = set()
    for subj, rel, obj in triples:
        if subj.startswith("ns:m.") or subj.startswith("ns:g."):
            entities.add(subj)
        if obj.startswith("ns:m.") or obj.startswith("ns:g."):
            entities.add(obj)
    return entities


def find_answer_variable(sparql):
    """
    Extract answer variable from SELECT clause.
    Usually ?x
    """
    match = re.search(r"SELECT DISTINCT (\?\w+)", sparql)
    if match:
        return match.group(1)
    return None


def find_reasoning_path(sparql):
    """
    Returns ordered path as:
    [
        (node_0, relation, direction, node_1),
        (node_1, relation, direction, node_2),
        ...
    ]
    """

    triples = extract_triples(sparql)
    if not triples:
        return None

    graph = build_graph(triples)
    answer_var = find_answer_variable(sparql)

    if answer_var is None:
        return None

    entity_constants = find_entity_constants(triples)
    if not entity_constants:
        return None

    for start in entity_constants:

        visited = set()
        queue = deque()
        queue.append((start, []))

        while queue:
            node, path = queue.popleft()

            if node == answer_var:
                return path

            if node in visited:
                continue

            visited.add(node)

            for neighbor, rel, direction in graph[node]:
                new_path = path + [(node, rel, direction, neighbor)]
                queue.append((neighbor, new_path))

    return None


def extract_hop_supervision(question_text, sparql):
    """
    Returns list of hop supervision tuples:
    [
        {
            "question": question_text,
            "relation": relation_name,
            "direction": +1 or -1
        },
        ...
    ]
    """

    path = find_reasoning_path(sparql)
    if path is None:
        return None

    hops = []
    for node, rel, direction, next_node in path:
        hops.append({
            "question": question_text,
            "relation": rel,
            "direction": direction
        })

    return hops