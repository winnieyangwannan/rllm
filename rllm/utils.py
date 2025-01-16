import time
from typing import List

import torch
import vertexai

from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from sentence_transformers import SentenceTransformer, util
from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory

def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = 'cloud-llm-test',
    location: str = "us-central1",
    model_id: str = "gemini-1.5-pro-002",
    retry_count: int = 1e9,
) -> List[str]:
    """
    Calls a Gemini LLM on Vertex AI to generate n responses at a given temperature.
    
    Args:
        prompt (str): The text prompt to send to the LLM.
        system_prompt (str): System instruction or system prompt to send to the model.
        n (int): Number of responses to generate.
        temperature (float): Sampling temperature.
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        retry_count (int): Number of times to retry on rate-limit errors.
    
    Returns:
        List[str]: A list of response texts from the Gemini model.
    """

    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)

    # Define which harm categories to allow (or set thresholds).
    HARM_CATEGORIES = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    # Instantiate the GenerativeModel
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    # Add an exponential backoff for rate limit errors
    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    for attempt in range(retry_count):
        try:
            # Request multiple candidates by specifying n (candidate_count)
            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=[
                    SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE)
                    for h in HARM_CATEGORIES
                ]
            )
            # Once successful, break out of the retry loop
            break
        except Exception as e:
            # Retry if there's a rate-limit error (HTTP 429)
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                print("Exception: ", e)
                return []  # or raise an exception if desired

    # Collect the texts from all returned candidates
    # Depending on the library version, this might need to be adjusted 
    # if the `response` shape is different
    try:
        [candidate.text for candidate in response.candidates]
        return [candidate.text for candidate in response.candidates]
    except Exception as e:
        print("Error extracting text from response:", e)
        return []

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

if __name__ == '__main__':
    print(call_gemini_llm('hello', 'You are freindly, be freindly', n=2))