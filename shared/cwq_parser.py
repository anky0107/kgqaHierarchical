import json
from utils.sparql_parser import extract_hop_supervision, extract_triples

class CWQParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def get_parsed_samples(self):
        samples = []
        for item in self.data:
            hops = extract_hop_supervision(item["question"], item["sparql"])
            triples = extract_triples(item["sparql"])
            samples.append({
                "id": item.get("ID", ""),
                "question": item["question"],
                "sparql": item["sparql"],
                "hops": hops,
                "triples": triples
            })
        return samples
