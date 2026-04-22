# Architectural Brainstorm: Beyond Tagged Multi-Tasking

The current bottleneck in our Universal Agent (Experiment 10) is **Vocabulary Dependence**. The model treats `film.film.directed_by` (Freebase) and `directed_by` (MetaQA) as completely independent classes in a 916-way classification head. 

To achieve true zero-shot generalization (like DRKG or UltraQuery), we must abandon the "Global Classification Head" and move towards **Vocabulary-Independent Reasoning**. 

Here are three out-of-the-box architectures for future experiments:

---

## 1. The "UltraQuery" Approach: Schema-Agnostic Projection

**Concept**: Instead of predicting a specific relation *name*, the network predicts a **vector in the embedding space**, and we pick the relation closest to that vector.

**How it works**:
1. We pre-train relation embeddings (using TransE or a frozen BERT) so that `film.film.directed_by` and `directed_by` are mathematically close to each other.
2. The question encoder (RoBERTa) takes "Who directed Inception?" and outputs a `target_vector`.
3. We calculate the Cosine Similarity between `target_vector` and **ONLY the relations connected to "Inception"** in the graph.
4. We execute the relation with the highest similarity.

**Why it’s powerful**: The model no longer has a 916-class output head. It just outputs vectors. It never needs to see a dataset tag, and it can generalize to a completely new KG (like Wikidata) on day one, as long as we can embed the new relations.

## 2. The "NS-KGQA" Approach: Neuro-Symbolic LLM Translation

**Concept**: Stop trying to force a small BERT model to learn graph topologies. Use an LLM to translate English into an abstract "Meta-Path", and let a symbolic engine do the rest.

**How it works**:
1. **Semantic Parsing**: A small LLM (e.g., Llama-3-8b or a heavily fine-tuned RoBERTa-Seq2Seq) translates the question into an abstract syntax: 
   `Query: "Who directed Inception?" -> Abstract Plan: FIND(Entity="Inception", Relation_Type="Creator")`
2. **Schema Alignment**: We use a fast similarity search (FAISS) to map the abstract `Relation_Type="Creator"` to whatever schema we are currently plugged into (`film.film.directed_by` for Freebase, `directed_by` for MetaQA).
3. **Execution**: Standard symbolic execution.

**Why it’s powerful**: It completely decouples the linguistic complexity of the question from the structural complexity of the specific Knowledge Graph.

## 3. The "STAGE" Approach: Subgraph Structural Prompting (PullNet 2.0)

**Concept**: Turn the KGQA task into a Reading Comprehension task over a localized graph.

**How it works**:
1. Before feeding the question to the model, we use the literal physical Knowledge Graph to extract the 2-hop neighborhood around the topic entity ("Inception").
2. We serialize this neighborhood into text: `[NODE] Inception [EDGE] directed_by [NODE] Christopher Nolan [EDGE] starring [NODE] Leonardo DiCaprio...`
3. We concatenate this literal graph text to the question: `Question: Who directed Inception? Graph Context: ...`
4. We train a Reader model (like Longformer or a local LLM) to simply point to the answer entity in the context string.

**Why it’s powerful**: This is what LLMs are best at. By serializing the subgraph into text during the forward pass, the model becomes 100% blind to which dataset the graph came from. It is purely grounded reasoning.

---

### Recommendation for Next Step
The **UltraQuery Projection Method (Idea 1)** is the most scientifically interesting transition from our current architecture. It keeps the hierarchical planning framework but replaces the rigid classification head with a fluid vector-similarity module, solving the "Language Dialect" problem permanently.
