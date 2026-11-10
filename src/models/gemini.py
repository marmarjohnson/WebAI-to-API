# src/models/gemini.py
import asyncio
from typing import Optional, List, Union
from pathlib import Path
from gemini_webapi import GeminiClient as WebGeminiClient
from app.logger import logger
from app.services.model_registry import registry, is_fallback_error, request_timeout


class MyGeminiClient:
    """
    Wrapper for the Gemini Web API client.

    All model names pass through the model registry, so a request for a model the
    account can no longer use (e.g. Pro after a subscription lapses) is served by
    the best model that still works instead of hanging or failing.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: str | None = None) -> None:
        self.client = WebGeminiClient(secure_1psid, secure_1psidts, proxy)

    async def init(self, **kwargs) -> None:
        await self.client.init(**kwargs)

    async def generate_content(self, message: str, model: str, files: Optional[List[Union[str, Path]]] = None):
        candidates = await registry.candidates(self.client, model)
        for i, name in enumerate(candidates):
            try:
                return await asyncio.wait_for(
                    self.client.generate_content(message, model=name or None, files=files),
                    timeout=request_timeout(),
                )
            except Exception as e:
                if not is_fallback_error(e) or i == len(candidates) - 1:
                    raise
                registry.mark_failed(name, e)
                logger.warning(f"Retrying on '{candidates[i + 1]}'")

    async def close(self) -> None:
        await self.client.close()

    def start_chat(self, model: str):
        return self.client.start_chat(model=model or None)
