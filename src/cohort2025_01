#%%
import importlib
import llm_helper
importlib.reload(llm_helper)

import requests 
from elasticsearch import Elasticsearch
from tqdm import tqdm
import tiktoken


#%%
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")

model = "mistral-large-latest"
client = Mistral(api_key=mistral_key)

#%%
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

#%%
for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

#%%    
documents[:5] 


# %%
es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = 'course-questions'

es_client.indices.create(index=index_name, body=index_settings)
# %%
for doc in tqdm(documents):
    es_client.index(index=index_name, body=doc)
    
# %%
##Question 2
llm_helper.elastic_search(es_client, index_name, 'How do execute a command on a Kubernetes pod?')[0]['_score']

# %%
##Question 3
llm_helper.elastic_search(es_client, index_name, 'How do copy a file to a Docker container?')[2]['_source']['question']
# %%
##Question 4
answer = llm_helper.rag(client, model, es_client, index_name, 'How do copy a file to a Docker container?')
print(answer)

# %%
##Question 5
query = 'How do copy a file to a Docker container?'
search_results = llm_helper.elastic_search(es_client, index_name, query, 3)
prompt = llm_helper.build_prompt(query, search_results)
encoding = tiktoken.encoding_for_model('gpt-4o')

len(encoding.encode(prompt))
# %%
