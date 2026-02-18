import requests
import json

# Get embedding
with open('C:\\Users\\shell\\.openclaw\\workspace\\temp_embed2.json', 'r') as f:
    embed_req = json.load(f)

resp = requests.post('http://localhost:8040/embed', json=embed_req)
vec = resp.json()['embeddings'][0]

# Search Qdrant
search_req = {
    "vector": vec,
    "limit": 3,
    "score_threshold": 0.3,
    "with_payload": True
}

resp2 = requests.post('http://localhost:8001/collections/mem_chunks/points/search', json=search_req)
results = resp2.json()['result']

print(f"Memories recalled: {len(results)}")
for r in results:
    print(f"  - [{r['score']:.2f}] {r['payload'].get('text', '')[:80]}...")

# Store a test memory
import time
store_req = {
    "points": [{
        "id": int(time.time()),
        "vector": vec,
        "payload": {
            "text": "Auto-memory cycle tested at 03:00 - cron job working",
            "kind": "fact",
            "scope": "global",
            "importance": 0.6
        }
    }]
}

resp3 = requests.put('http://localhost:8001/collections/mem_chunks/points', json=store_req)
print(f"\nStored: {resp3.json()['status']}")

# Get updated count
resp4 = requests.get('http://localhost:8001/collections/mem_chunks')
print(f"Total memories: {resp4.json()['result']['points_count']}")
