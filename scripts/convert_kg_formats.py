import torch
import pickle
import lmdb
import os
import json
import time

def convert_kg_formats(input_path):
    print(f"Loading original KG from {input_path}...")
    start = time.time()
    kg = torch.load(input_path, map_location='cpu')
    print(f"Loaded in {time.time() - start:.2f}s")
    
    # 1. Save as Pickle Protocol 5 (Optimized RAM load)
    pickle_path = input_path.replace('.pt', '_v5.pkl')
    print(f"Converting to Pickle Protocol 5: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(kg, f, protocol=5)
    
    # 2. Save as LMDB (Lightning Memory-Mapped Database - Disk based)
    lmdb_path = input_path.replace('.pt', '_lmdb')
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    
    print(f"Converting to LMDB: {lmdb_path}")
    # Estimate map size (4GB input -> 8GB map size to be safe)
    env = lmdb.open(lmdb_path, map_size=10 * 1024 * 1024 * 1024) 
    
    with env.begin(write=True) as txn:
        # We store forward and backward as separate entries or separate databases
        # Here we'll store them as keys: "f:mid" and "b:mid"
        
        print("  Processing forward edges...")
        for mid, neighbors in kg['forward'].items():
            # Serialize the list of tuples to bytes
            txn.put(f"f:{mid}".encode('utf-8'), pickle.dumps(neighbors))
            
        print("  Processing backward edges...")
        for mid, neighbors in kg['backward'].items():
            txn.put(f"b:{mid}".encode('utf-8'), pickle.dumps(neighbors))
            
    env.close()
    print("Conversion complete!")

if __name__ == "__main__":
    KG_PATH = 'data/processed_kg/augmented_kg.pt'
    if os.path.exists(KG_PATH):
        convert_kg_formats(KG_PATH)
    else:
        print(f"Error: {KG_PATH} not found.")
