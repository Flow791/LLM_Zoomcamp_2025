#%%
import importlib
import llm_helper
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import os
from mistralai import Mistral
from dotenv import load_dotenv

importlib.reload(llm_helper)

# %%
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")

model = "mistral-large-latest"
client = Mistral(api_key=mistral_key)

qdrant_client = QdrantClient(url='http://localhost:6333')
embedding_model_name = 'BAAI/bge-small-en'
embedding_model = TextEmbedding(model_name=embedding_model_name)
collection_name = 'llm_zoomcamp_cohort_2'
EMBEDDING_DIMENSIONALITY = 384

# %%
question = 'I just discovered the course. Can I join now?'
llm_helper.rag(client, model, qdrant_client, collection_name, embedding_model_name, question)
# %%
question = 'I just discovered the course. Is it too late to join?'
llm_helper.rag(client, model, qdrant_client, collection_name, embedding_model_name, question)
# %%
