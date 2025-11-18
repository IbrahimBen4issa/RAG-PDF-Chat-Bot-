import chromadb
from chromadb.config import Settings
c = chromadb.Client(settings=Settings(is_persistent=True, persist_directory='chroma'))
col = c.get_collection('documents')
items = col.get(include=['metadatas','documents'], limit=50)
print('Raw items keys:', list(items.keys()))
print('Items repr (truncated):')
print(repr(items)[:2000])
metas_all = items.get('metadatas')
print('\nmetadatas structure:', type(metas_all))
if metas_all:
    # chromadb returns lists of lists per query; handle safely
    try:
        metas = metas_all[0]
    except Exception:
        metas = metas_all
    print('Sample metadata count:', len(metas))
    count_table = sum(1 for m in metas if isinstance(m, dict) and m.get('is_table'))
    print('Table chunks in sample:', count_table)
    for m in metas:
        if isinstance(m, dict) and m.get('is_table'):
            print('\nSample table preview:\n')
            print(m.get('table_preview'))
            break
    else:
        print('\nNo table chunks in sample')
else:
    print('No metadatas returned')
