import json

nb = json.load(open(r'c:\Users\swoop\dev\res\kgqa\colabNotebooks\cwqAnalysis.ipynb', 'r', encoding='utf-8'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'dropbox.com' in src and 'urls' in src:
        print("FOUND URLS CELL:")
        print(src)
        break
