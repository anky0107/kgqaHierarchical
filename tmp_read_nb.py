import json, sys

nb = json.load(open(r'c:\Users\swoop\dev\res\kgqa\colabNotebooks\cwqAnalysis.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
print(f"Total cells: {len(cells)}")
print("="*80)

for i, c in enumerate(cells):
    ct = c.get('cell_type', 'unknown')
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    
    out_text = ""
    if 'outputs' in c:
        for o in c['outputs']:
            if 'text' in o:
                ot = ''.join(o['text']) if isinstance(o['text'], list) else o['text']
                out_text += ot
            elif 'data' in o and 'text/plain' in o['data']:
                ot = ''.join(o['data']['text/plain']) if isinstance(o['data']['text/plain'], list) else o['data']['text/plain']
                out_text += ot
    
    print(f"\n--- Cell {i} ({ct}) ---")
    try:
        print(src[:600].encode('ascii', 'replace').decode('ascii'))
    except:
        print("[encoding error in source]")
    if out_text:
        try:
            print(f"\n  OUTPUT: {out_text[:500].encode('ascii', 'replace').decode('ascii')}")
        except:
            print("  [encoding error in output]")
    print()
