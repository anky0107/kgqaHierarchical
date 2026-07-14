"""
filter_freebase.py — Extract relevant subgraph from raw Freebase RDF
====================================================================

Overview
--------
The raw Freebase RDF dump is >30GB compressed. This script streams the 
compressed RDF file and extracts only the triples where either the subject 
or the object belongs to the set of MIDs appearing in the CWQ dataset.

Usage
-----
    python scripts/filter_freebase.py

Note: This is a one-time preprocessing step. The output TSV is subsequently 
ingested by `shared/kg_loader.py` to build the LMDB graph.
"""
# ──────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────
import gzip
import os
import time

# ──────────────────────────────────────────────────────
#  Extraction logic
# ──────────────────────────────────────────────────────

def filter_freebase(gz_path, mids_path, output_path):
    # Load MIDs
    print(f"Loading MIDs from {mids_path}...")
    with open(mids_path, 'r') as f:
        mids = set(line.strip() for line in f)
    
    # Pre-process MIDs to match RDF format: <http://rdf.freebase.com/ns/m.01234>
    rdf_mids = set(f"<http://rdf.freebase.com/ns/{mid}>" for mid in mids)
    
    print(f"Filtering {gz_path}...")
    start_time = time.time()
    count = 0
    kept = 0
    
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            count += 1
            if count % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {count//1000000}M lines... Kept: {kept:,} | Speed: {count/elapsed:.0f} lines/sec")
            
            # Fast check: see if any known MID string is in the line
            # This is much faster than full parsing for 2B lines
            # Most Freebase lines follow the format: <subj> <pred> <obj> .
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            subj = parts[0]
            obj = parts[2]
            
            if subj in rdf_mids or obj in rdf_mids:
                f_out.write(line)
                kept += 1

    print(f"\nFinished!")
    print(f"Total lines: {count:,}")
    print(f"Total kept: {kept:,}")
    print(f"Time taken: {(time.time() - start_time)/3600:.2f} hours")

# ──────────────────────────────────────────────────────
#  Main Execution Block
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    GZ_PATH = 'freebase-rdf-latest.gz'
    MIDS_PATH = 'data/cwq_mids.txt'
    OUTPUT_PATH = 'data/cwq_filtered_kg.tsv'
    
    if os.path.exists(GZ_PATH):
        filter_freebase(GZ_PATH, MIDS_PATH, OUTPUT_PATH)
    else:
        print(f"Error: {GZ_PATH} not found.")
