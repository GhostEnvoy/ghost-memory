import requests

# Get embedding for the query
query = 'memories categories what are they about'
r = requests.post('http://localhost:8040/embed', json={'texts': [query]})
embedding = r.json()['embeddings'][0]
print(f'Got embedding of dimension {len(embedding)}')

# Search Qdrant with lower threshold
q = requests.post('http://localhost:8001/collections/mem_chunks/points/search', json={
    'vector': embedding,
    'limit': 5,
    'score_threshold': 0.5,
    'with_payload': True
})
results = q.json()['result']
print(f'Found {len(results)} memories above threshold 0.5')
for mem in results:
    text = mem.get('payload', {}).get('text', '')[:60]
    score = mem.get('score', 0)
    print(f'  [{score:.2f}] {text}')
