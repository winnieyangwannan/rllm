import asyncio
import aiohttp
from typing import List, Dict, Any
from dataclasses import dataclass
import random

@dataclass
class Endpoint:
    url: str
    weight: float = 1.0
    current_load: int = 0

class DistributedLLMClient:
    def __init__(self, endpoints: List[Dict[str, Any]]):
        """
        Initialize with a list of endpoint configurations.
        endpoints: List of dicts with {'url': str, 'weight': float}
        """
        self.endpoints = [
            Endpoint(**endpoint_config)
            for endpoint_config in endpoints
        ]
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _select_endpoint(self) -> Endpoint:
        # Simple weighted load balancing
        available = [ep for ep in self.endpoints if ep.current_load < 100]  # arbitrary limit
        if not available:
            raise RuntimeError("All endpoints are at capacity")
        
        # Weight by both configured weight and inverse of current load
        weights = [
            ep.weight * (1.0 / (ep.current_load + 1))
            for ep in available
        ]
        return random.choices(available, weights=weights)[0]

    async def _make_request(self, endpoint: Endpoint, messages: List[Dict], sampling_params: Dict) -> Dict:
        endpoint.current_load += 1
        try:
            payload = {
                "messages": messages,
                **sampling_params
            }
            async with self.session.post(endpoint.url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        finally:
            endpoint.current_load -= 1

    async def chat(self, messages: List[List[Dict]], sampling_params: Dict) -> List[Dict]:
        """
        Process a batch of message lists in parallel across available endpoints.
        """
        async def process_one(messages):
            endpoint = self._select_endpoint()
            return await self._make_request(endpoint, messages, sampling_params)

        tasks = [
            process_one(msg) 
            for msg in messages
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def create_client(endpoints: List[Dict[str, Any]]) -> 'DistributedLLMClient':
        """Factory method to create and initialize the client"""
        client = DistributedLLMClient(endpoints)
        await client.__aenter__()
        return client