import requests
import json
import time
from datetime import datetime

# User's latest message (voice note transcription)
user_message = "Alright so you say you have around a thousand six hundred memories or what are all these memories about what we spoke about are these memories about me are their categories or something or is it just one mush of data"

# Assistant's response 
assistant_response = "Explored Qdrant, found memories with categories like 'unknown' and 'fact', showed sample"

# Create embedding for the exchange
r = requests.post('http://localhost:8040/embed', json={'texts': [user_message + " " + assistant_response]})
embedding = r.json()['embeddings'][0]

# Store to Qdrant with ID
timestamp = datetime.utcnow().isoformat() + 'Z'
point_id = int(time.time() * 1000)  # timestamp as ID
point = {
    'id': point_id,
    'vector': embedding,
    'payload': {
        'text': f"User asked about memories: {user_message[:80]}... | Response: explained memories stored in Qdrant with categories",
        'kind': 'fact',
        'importance': 0.6,
        'timestamp': timestamp,
        'source': 'auto-memory-cron'
    }
}

q = requests.put('http://localhost:8001/collections/mem_chunks/points', json={
    'points': [point]
})

print(f'Stored exchange: {q.status_code}')
if q.status_code == 200:
    print('Success!')
else:
    print(f'Response: {q.json()}')
