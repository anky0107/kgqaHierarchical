import json
import os

os.makedirs('data', exist_ok=True)
nb = json.load(open(r'c:\Users\swoop\dev\res\kgqa\colabNotebooks\cwqAnalysis.ipynb', 'r', encoding='utf-8'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'urls =' in src and 'requests.get' in src:
        # write exactly this cell to download_cwq.py
        with open('data/download_cwq.py', 'w', encoding='utf-8') as f:
            f.write(src)
        print("Wrote data/download_cwq.py")
        break
