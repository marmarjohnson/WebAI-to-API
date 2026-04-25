# src/models/gemini.py
from typing import Optional, List, Union
from pathlib import Path
from gemini_webapi import GeminiClient as WebGeminiClient
from app.config import CONFIG

class MyGeminiClient:
    """
    Wrapper for the Gemini Web API client.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: str | None = None) -> None:
        self.client = WebGeminiClient(secure_1psid, secure_1psidts, proxy)

    async def init(self, **kwargs) -> None:
        await self.client.init(**kwargs)

    async def generate_content(self, message: str, model: str, files: Optional[List[Union[str, Path]]] = None):
        return await self.client.generate_content(message, model=model, files=files)

    async def close(self) -> None:
        await self.client.close()

    def start_chat(self, model: str):
        return self.client.start_chat(model=model)
