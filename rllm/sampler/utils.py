import gc
import requests
import time
import psutil
from typing import List

from openai import ChatCompletion
import torch

from rllm.sampler.sampler_types import Sample

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
def convert_openai_response_to_samples(response: ChatCompletion) -> List[Sample]:
    choices = response.choices
    samples = []
    for choice in choices:
        response = choice.message.content
        choice_log_probs = choice.logprobs.content
        log_probs = []
        for token_log_prob in choice_log_probs:
            log_probs.append(token_log_prob.logprob)
        mask = [True for _ in log_probs]
        # response_tokens = tokenizer.encode(response)
        sample =  Sample(
            response=response,
            tokens=None,
            mask=mask,
            log_probs=log_probs,
            reward=None,
            is_correct=None,
        )
        samples.append(sample)
    return samples


def kill_process_and_children(pid: int) -> bool:
    """Kill a process and all its children processes.
    
    Args:
        pid: Process ID to kill
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            
        # Kill parent
        parent.kill()
        
        # Wait for processes to terminate
        psutil.wait_procs(children + [parent], timeout=3)
        
        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def check_server_health(port: int, timeout: int = 5) -> bool:
    """Check if server at given port is healthy"""
    try:
        response = requests.get(f"http://0.0.0.0:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def wait_for_server(port: int, timeout: int = 1800, interval: int = 5):
    """Wait for server to be healthy"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_server_health(port):
            return True
        time.sleep(interval)
    raise TimeoutError(f"Server on port {port} failed to start within {timeout} seconds")
