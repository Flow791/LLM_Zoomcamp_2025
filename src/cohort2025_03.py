#%%
import requests
import pandas as pd
from minsearch import Index, VectorSearch
import uuid
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

from qdrant_client import QdrantClient, models
from rouge import Rouge


#%%
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

#%%
documents[0]
#%%
ground_truth[0]

#%%
from tqdm.auto import tqdm

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }
    
def cosine(u, v):
    u_norm = np.sqrt(u.dot(u))
    v_norm = np.sqrt(v.dot(v))
    return u.dot(v) / (u_norm * v_norm)


# %%
#Question 1
text_fields = ['question', 'section', 'text']
keyword_fields = ['course', 'id']
boost = {'question': 1.5, 'section': 0.1}

index = Index(
    text_fields=text_fields,
    keyword_fields=keyword_fields
)

index.fit(documents)

def search_func(q):
    return index.search(
        query=q['question'], 
        filter_dict={'course': q['course']},
        boost_dict=boost,
        num_results=5
    )

# %%
evaluate(ground_truth, search_func)

# %%
#Question 2
texts = []

for doc in documents:
    texts.append(doc['question'])
  
def get_X_from_texts(texts):  
    pipeline = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )

    X = pipeline.fit_transform(texts)
    return pipeline, X

pipeline, X = get_X_from_texts(texts)

vector_index = VectorSearch(
    keyword_fields={'course'}
)
vector_index.fit(X, documents)

def vector_search(q):
    vector_query = pipeline.transform([q['question']])
    
    return vector_index.search(
        query_vector=vector_query,
        filter_dict={'course': q['course']},
        num_results=5
    )

# %%
evaluate(ground_truth, vector_search)

# %%
#Question 3
complete_texts = []
for doc in documents:
    complete_texts.append(doc['question'] + ' ' + doc['text'])
    
pipeline_complete, X_complete = get_X_from_texts(complete_texts)

vector_index_complete = VectorSearch(
    keyword_fields={'course'}
)
vector_index_complete.fit(X_complete, documents)

def vector_search_complete(q):
    vector_query = pipeline_complete.transform([q['question']])
    
    return vector_index_complete.search(
        query_vector=vector_query,
        filter_dict={'course': q['course']},
        num_results=5
    )

# %%
evaluate(ground_truth, vector_search_complete)

#%%
documents[:1]

# %%
#Question 4:
client_qdrant = QdrantClient(url='http://localhost:6333')
client_qdrant.delete_collection("cohort_hw_3")

client_qdrant.create_collection(
        collection_name='cohort_hw_3',
        vectors_config={
            "jina-small": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
        }
    )

client_qdrant.upsert(
    collection_name="cohort_hw_3",
    points=[
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "jina-small": models.Document(
                    text=doc['question'] + ' ' + doc["text"],
                    model="jinaai/jina-embeddings-v2-small-en",
                )
            },
            payload={
                "id": doc["id"],
                "section": doc["section"],
                "text": doc["text"],
                "question": doc["question"],
                "course": doc["course"],
            }
        )
        for doc in documents
    ]
)
# %%
def qdrant_search(q):
    results = client_qdrant.query_points(
        collection_name="cohort_hw_3",
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=q['question'],
                    model="jinaai/jina-embeddings-v2-small-en",
                ),
                using="jina-small",
                limit=(10 * 5),
            ),
        ],
        query=models.Document(
            text=q['question'],
            model="jinaai/jina-embeddings-v2-small-en",
        ),
        using="jina-small",
        limit=5,
        with_payload=True,
    )
    
    payloads = []
    for points in results.points:
        payloads.append(points.payload)
    
    return payloads

# %%
score = evaluate(ground_truth, qdrant_search)
# %%
score

# %%
#Question 5:
results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
df_results = pd.read_csv(results_url)

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)

pipeline.fit(df_results.answer_llm + ' ' + df_results.answer_orig + ' ' + df_results.question)

def compute_similarity(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = pipeline.transform([answer_llm])[0]
    v_orig = pipeline.transform([answer_orig])[0]
    
    return cosine(v_llm, v_orig)

similarity = []
for record in tqdm(df_results.to_dict(orient='records')):
    sim = compute_similarity(record)
    similarity.append(sim)

# %%
df_results['similarity'] = similarity
df_results['similarity'].mean()

# %%
#Question 6:
rouge_scorer = Rouge()

def compute_rouge_1_f1(record):
    scores = rouge_scorer.get_scores(record['answer_llm'], record['answer_orig'])[0]
    return scores['rouge-1']['f']

rouge_f1_scores = [
    compute_rouge_1_f1(record)
    for _, record in tqdm(df_results.iterrows(), total=len(df_results))
]
# %%
df_results['rouge_1_f1'] = rouge_f1_scores
df_results['rouge_1_f1'].mean()
# %%
