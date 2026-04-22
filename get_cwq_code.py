import json

nb = json.load(open(r'path/to/colabNotebooks/cwqAnalysis.ipynb', 'r', encoding='utf-8'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'dropbox.com' in src and 'urls' in src:
        print("FOUND URLS CELL:")
        print(src)
        break
