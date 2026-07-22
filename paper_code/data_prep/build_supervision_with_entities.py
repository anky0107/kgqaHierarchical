import json
import torch
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sparql_parser import find_reasoning_path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_domain(rel):
    return rel.split(".")[0]


def extract_split(cwq_data):

    hop_samples = []
    relation_set = set()
    domain_set = set()
    entity_set = set()

    for item in cwq_data:

        question = item["question"]
        sparql = item["sparql"]

        path = find_reasoning_path(sparql)
        if path is None:
            continue

        for node, rel, direction, next_node in path:

            relation_set.add(rel)
            domain_set.add(get_domain(rel))
            entity_set.add(node)

            hop_samples.append({
                "question": question,
                "entity": node,
                "relation": rel,
                "domain": get_domain(rel)
            })

    return hop_samples, relation_set, domain_set, entity_set


def main():

    train_data = load_json("data/cwq_train.json")
    dev_data = load_json("data/cwq_dev.json")
    test_data = load_json("data/cwq_test.json")

    train_hops, train_rel, train_dom, train_ent = extract_split(train_data)
    dev_hops, dev_rel, dev_dom, dev_ent = extract_split(dev_data)
    test_hops, test_rel, test_dom, test_ent = extract_split(test_data)

    global_rel = train_rel | dev_rel | test_rel
    global_dom = train_dom | dev_dom | test_dom
    global_ent = train_ent | dev_ent | test_ent

    print("Relations:", len(global_rel))
    print("Domains:", len(global_dom))
    print("Entities:", len(global_ent))

    relation2id = {r: i for i, r in enumerate(sorted(global_rel))}
    domain2id = {d: i for i, d in enumerate(sorted(global_dom))}
    entity2id = {e: i for i, e in enumerate(sorted(global_ent))}

    relation_to_domain = torch.zeros(len(relation2id), dtype=torch.long)

    for r, r_id in relation2id.items():
        relation_to_domain[r_id] = domain2id[get_domain(r)]

    def build_split(hops):
        questions = []
        entities = []
        relations = []
        domains = []

        for sample in hops:
            questions.append(sample["question"])
            entities.append(entity2id[sample["entity"]])
            relations.append(relation2id[sample["relation"]])
            domains.append(domain2id[sample["domain"]])

        return questions, torch.tensor(entities), torch.tensor(relations), torch.tensor(domains)

    os.makedirs("data/processed_entity", exist_ok=True)

    train_q, train_e, train_r, train_d = build_split(train_hops)
    dev_q, dev_e, dev_r, dev_d = build_split(dev_hops)
    test_q, test_e, test_r, test_d = build_split(test_hops)

    torch.save(relation2id, "data/processed_entity/relation2id.pt")
    torch.save(domain2id, "data/processed_entity/domain2id.pt")
    torch.save(entity2id, "data/processed_entity/entity2id.pt")
    torch.save(relation_to_domain, "data/processed_entity/relation_to_domain.pt")

    torch.save(train_e, "data/processed_entity/train_entities.pt")
    torch.save(train_r, "data/processed_entity/train_relations.pt")
    torch.save(train_d, "data/processed_entity/train_domains.pt")

    torch.save(dev_e, "data/processed_entity/dev_entities.pt")
    torch.save(dev_r, "data/processed_entity/dev_relations.pt")
    torch.save(dev_d, "data/processed_entity/dev_domains.pt")

    torch.save(test_e, "data/processed_entity/test_entities.pt")
    torch.save(test_r, "data/processed_entity/test_relations.pt")
    torch.save(test_d, "data/processed_entity/test_domains.pt")

    # Save raw question lists for embedding
    torch.save(train_q, "data/processed_entity/train_questions_raw.pt")
    torch.save(dev_q, "data/processed_entity/dev_questions_raw.pt")
    torch.save(test_q, "data/processed_entity/test_questions_raw.pt")


if __name__ == "__main__":
    main()