from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

import httpx
from openai import OpenAI


class LLM_Model:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        rits_api_key: str ="EMPTY",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        # Store configuration for creating thread-safe clients
        self.api_key = api_key
        self.base_url = base_url
        self.rits_api_key = rits_api_key
        self.timeout = timeout
        
        self.llm_config = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _get_client(self):
        """Create a new OpenAI client for thread-safe operations"""
        # Use a fresh HTTP client with limits to prevent connection pool issues
        http_client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=60.0),
            trust_env=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
            max_retries=3,
            timeout=1800.0,
            default_headers={
                "RITS_API_KEY": self.rits_api_key,
            },
        )

    def infer(self, messages):
        # Create a new client for each request to ensure thread safety
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                messages=messages,
                **self.llm_config,
            )
            return response.choices[0].message.content
        finally:
            # Properly close both the client and its HTTP client
            try:
                if hasattr(client, '_client') and client._client:
                    client._client.close()
                client.close()
            except Exception:
                pass  # Ignore cleanup errors



def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
