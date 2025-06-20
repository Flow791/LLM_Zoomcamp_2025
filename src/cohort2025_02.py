#%%
import importlib
import llm_helper

importlib.reload(llm_helper)

from qdrant_client import QdrantClient, models
from tqdm import tqdm
import requests
import json
from fastembed import TextEmbedding
import numpy as np

# %%
qdrant_client = QdrantClient(url='http://localhost:6333')
EMBEDDING_DIMENSIONALITY = 512

# %%
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
# %%
documents_raw
# %%
for model in TextEmbedding.list_supported_models():
    if model["dim"] == EMBEDDING_DIMENSIONALITY:
        print(json.dumps(model, indent=2))
# %%
embedding_model_name = 'jinaai/jina-embeddings-v2-small-en'
collection_name = 'llm_zoomcamp_rag'

#%%
llm_helper.create_qdrant_collection(qdrant_client, EMBEDDING_DIMENSIONALITY, collection_name)
# %%
llm_helper.insert_vectors_to_qdrant(documents_raw, embedding_model_name, collection_name)

# %%
#Test the search function
question = 'I just discovered the course. Can I join now?'
answer = llm_helper.qdrant_search(qdrant_client, collection_name, embedding_model_name, question)

answer.points[0].payload['text']

# %%
##Question 1:
embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-small-en")
embedded_question = np.array(list(embedding_model.embed([question])))[0]

embedded_question.min()

# %%
print(np.linalg.norm(embedded_question))
print(embedded_question.dot(embedded_question))

# %%
##Question 2:
doc = 'Can I still join the course after the start date?'
embedded_doc = np.array(list(embedding_model.embed(doc)))[0]

print(f'{len(embedded_doc)} - {len(embedded_question)}')
embedded_question.dot(embedded_doc)


#%%
documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]

# %%
##Question 3:
embedded_texts = []

for doc in documents:
    embedded_text = np.array(list(embedding_model.embed([doc['text']])))[0]
    embedded_texts.append(embedded_text)
    
embedded_texts = np.array(embedded_texts)
embedded_texts.dot(embedded_question)

# %%
##Question 4:
full_embedded_texts = []

for doc in documents:
    full_embedded_text = np.array(list(embedding_model.embed([doc['question'] + ' ' + doc['text']])))[0]
    full_embedded_texts.append(full_embedded_text)
    
full_embedded_texts = np.array(full_embedded_texts)
full_embedded_texts.dot(embedded_question)

#Position is better now beacaus of the question equality. As the question is the same, vector will now be more similar to the question vector.

# %%
##Question 5:
models = sorted(TextEmbedding.list_supported_models(), key=lambda x: x['dim'])

for model in models:
    print(f"Model: {model['model']}, Dimensionality: {model['dim']}")

# %%
#Question 6: Prep
embedding_model_name = 'BAAI/bge-small-en'
embedding_model = TextEmbedding(model_name=embedding_model_name)
collection_name = 'llm_zoomcamp_cohort_2'
EMBEDDING_DIMENSIONALITY = 384


# %%
##Question 6: 
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()


documents = []

for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

llm_helper.create_qdrant_collection(qdrant_client, EMBEDDING_DIMENSIONALITY, collection_name)
llm_helper.insert_vectors_to_qdrant(qdrant_client, documents, embedding_model_name, collection_name, True)

# %%
llm_helper.qdrant_search(qdrant_client, collection_name, embedding_model_name, 'I just discovered the course. Can I join now?')
