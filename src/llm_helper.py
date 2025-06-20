from qdrant_client import QdrantClient, models

def elastic_search(es_client, index_name, query, size=3):
    search_query = {
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit)
        
    return result_docs

def build_prompt(query, search_results):
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT:
        {context}
        """.strip()

    context = ''

    for doc in search_results:
        doc_source = doc['_source']
        context = context + f"""
        Q: {doc_source["question"]}
        A: {doc_source["text"]}
        """.strip() + "\n\n"
        
    prompt = prompt_template.format(question=query, context=context)
    return prompt

def llm(llm_client, model, prompt):
    chat_response = llm_client.chat.complete(
        model=model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return chat_response.choices[0].message.content

def rag(llm_client, model, es_client, index_name, query):
    search_results = elastic_search(es_client, index_name, query, 3)
    prompt = build_prompt(query, search_results)
    print(f'Prompt length: {len(prompt)}')
    answer = llm(llm_client, model, prompt)
    
    return answer


def create_qdrant_collection(qdrant_client, dimensionality, collection_name):
    """
    Create a Qdrant collection with the specified name and embedding model.
    """
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimensionality,
            distance=models.Distance.COSINE
        )
    )

def create_qdrant_point(embedding_model, doc_text, course, id):
    point = models.PointStruct(
                id=id,
                vector=models.Document(text=doc_text, model=embedding_model), #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
                payload={
                    "text": course['text'],
                    "section": course['section'],
                    "course": course['course']
                } #save all needed metadata fields
            )
    return point
    
def insert_vectors_to_qdrant(qdrant_client, documents_raw, embedding_model, collection_name, is_alter = False):
    """
    Insert vectors into Qdrant collection from the provided documents.
    """
    points = []
    id = 0

    for course in documents_raw:
            if is_alter:
                doc_text = course['question'] + ' ' + course['text']
                point = create_qdrant_point(embedding_model, doc_text, course, id)
                points.append(point)
                id += 1
                
            else:
                for doc in course['documents']:
                    doc_text = doc['text']
                    point = create_qdrant_point(embedding_model, doc_text, course, id)
                    points.append(point)
                    id += 1
                
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    
def qdrant_search(qdrant_client, collection_name, embedding_model, query, limit=1):
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text=query,
            model=embedding_model
        ),
        limit=limit,
        with_payload=True
    )
    
    return results