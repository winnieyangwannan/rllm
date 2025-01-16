import time
from typing import List

import torch
import vertexai

from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from sentence_transformers import SentenceTransformer, util
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory

def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    project_id: str = 'cloud-llm-test',
    location: str = "us-central1",
    model_id: str = "gemini-1.5-pro-002",
    retry_count: int = 1e9,
) -> str:
    """
    Calls a Gemini LLM on Vertex AI to generate a response.
    
    Args:
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        prompt (str): The text prompt to send to the LLM.
    
    Returns:
        str: The generated text from the Gemini model.
    """

    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)
    HARM_CATEGORIES = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[
            system_prompt
        ],
    )
    backoff = 1
    retry_count = int(retry_count)
    for _ in range(retry_count):
        try:
            response = model.generate_content([prompt], safety_settings=[
                    SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE)
                    for h in HARM_CATEGORIES
                ])
            break
        except Exception as e:
            # If error has string 429 in it, it's a rate limit error
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            else:
                print(e)
                return None
    # text is not part of response
    try:
        response.text
        return response.text
    except:
        return None


class RAG:

    def __init__(self, docs: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)
    
    def top_k(self, query, k=1):
        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'score': score,
                'text': self.docs[int(idx)]
            })
        return results
