import requests

# Check what's in Qdrant
resp = requests.post('http://localhost:8001/collections/memories/points/search', 
    json={'vector': [0.0] * 768, 'limit': 10}, timeout=30)
print(f'Search status: {resp.status_code}')
results = resp.json()
print(f'Results: {results.get("result", [])}')
print(f'Total: {results.get("status", "")}')
