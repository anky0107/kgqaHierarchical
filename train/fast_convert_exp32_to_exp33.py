import json, re, os

ROOT = r"C:\Users\swoop\dev\res\kgqa\kgqaHierarchical"
in_path = os.path.join(ROOT, "data/exp32_t5_cot_train.json")
out_path = os.path.join(ROOT, "data/exp33_t5_pointer_train.json")

print(f"Loading {in_path}...")
with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for item in data:
    prompt = item["prompt"]
    target = item["target"]
    q = item["question"]
    
    # Replace instruction in prompt
    old_inst = "Which of the above candidates is the correct answer to the question? Reason through the relations and output the answer."
    new_inst = "Which of the above candidates is the correct answer to the question? Output the Candidate Index and reason through the relations."
    prompt = prompt.replace(old_inst, new_inst)
    
    # Extract the target entity
    # Old target: Reasoning: The relations involved are {path}. Therefore the correct entity is {name}. Answer: {name}
    # Or: Reasoning: This entity directly matches the question constraints. Answer: {name}
    
    m = re.search(r"Answer:\s*(.+)$", target)
    if not m:
        continue
    gold_name = m.group(1).strip()
    
    # Extract the reasoning path
    m_path = re.search(r"Reasoning: (.*?)(?: Therefore the correct entity is| Answer:)", target)
    if m_path:
        reasoning = m_path.group(1).strip()
    else:
        reasoning = "matches constraints directly"
        
    if "The relations involved are " in reasoning:
        reasoning = reasoning.replace("The relations involved are ", "")
        
    # Find the gold name's index in the prompt
    lines = prompt.split("\n")
    gold_idx = None
    for line in lines:
        m_cand = re.match(r"^(\d+)\.\s+(.*?)(?:\s+\(Path:.*)?$", line)
        if m_cand:
            idx = int(m_cand.group(1))
            name = m_cand.group(2).strip()
            if name == gold_name:
                gold_idx = idx
                break
                
    if gold_idx is None:
        continue
        
    new_target = f"Candidate Index: [{gold_idx}] | Reasoning Path: {reasoning}"
    
    new_data.append({
        "prompt": prompt,
        "target": new_target,
        "question": q
    })

print(f"Converted {len(new_data)} samples.")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f)
print(f"Saved to {out_path}.")
