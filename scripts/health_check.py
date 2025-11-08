# scripts/health_check.py
from src.ingest.chroma_client import col

def _flatten(groups):
    return [x for g in (groups or []) for x in g]

def scan(threshold=3, page=5000):
    """
    Detecta documentos basura (<= threshold chars) con paginación.
    """
    offset = 0
    bad_ids = []
    while True:
        res = col.get(include=["documents"], limit=page, offset=offset)
        ids_groups  = res.get("ids") or []
        docs_groups = res.get("documents") or []
        if not ids_groups:
            break

        for ids, docs in zip(ids_groups, docs_groups):
            for i, d in zip(ids, docs):
                if isinstance(d, str) and len(d.strip()) <= threshold:
                    bad_ids.append(i)

        if sum(len(g) for g in ids_groups) < page:
            break
        offset += page

    # dedup manteniendo orden
    seen = set()
    bad_ids = [i for i in bad_ids if not(i in seen or seen.add(i))]
    print(f"Total documentos basura detectados (<= {threshold} chars): {len(bad_ids)}")
    return bad_ids

def clean(threshold=3, batch=100):
    bad_ids = scan(threshold=threshold)
    if not bad_ids:
        print("✨ Nada que borrar. La VDB está limpia.")
        return
    for i in range(0, len(bad_ids), batch):
        col.delete(ids=bad_ids[i:i+batch])
    print(f"✅ Borrados {len(bad_ids)} documentos basura.")

if __name__ == "__main__":
    import sys
    if "--clean" in sys.argv:
        clean()
    else:
        scan()
