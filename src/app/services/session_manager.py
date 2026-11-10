import asyncio
from typing import Dict, NamedTuple
from app.logger import logger
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.model_registry import registry, is_fallback_error, request_timeout


class SessionModelUnavailable(Exception):
    """An existing conversation's model is no longer usable.

    Its history lives in a Gemini thread bound to that model, so silently continuing on
    another model would drop context. Callers that hold the full message history
    (e.g. /v1/chat/completions) catch this and rebuild a fresh session.
    """


class _Session(NamedTuple):
    chat: object
    requested: str  # canonical model the client asked for
    used: str       # model actually serving this conversation (may be a downgrade)


class SessionManager:
    def __init__(self, client):
        self.client = client
        self.sessions: Dict[str, _Session] = {}
        self.lock = asyncio.Lock()

    async def _send(self, chat, message, images):
        return await asyncio.wait_for(chat.send_message(prompt=message, files=images), timeout=request_timeout())

    async def get_response(self, model, message: str, session_id: str = "default", images=None):
        async with self.lock:
            wanted, candidates = await registry.resolve(self.client.client, model)
            existing = self.sessions.get(session_id)

            if existing and existing.requested == wanted:
                if existing.used not in candidates:
                    self.sessions.pop(session_id, None)
                    raise SessionModelUnavailable(
                        f"Model '{existing.used}' for session {session_id} is no longer available"
                    )
                try:
                    return await self._send(existing.chat, message, images)
                except Exception as e:
                    logger.error(f"Error in session {session_id} get_response: {e}", exc_info=True)
                    self.sessions.pop(session_id, None)
                    if is_fallback_error(e):
                        registry.mark_failed(existing.used, e)
                    raise

            if existing:
                logger.info(f"Model changed for session {session_id}: {existing.requested} → {wanted}, restarting session")
                self.sessions.pop(session_id, None)

            # New conversation: no history to lose, so walk down the fallback chain.
            for i, name in enumerate(candidates):
                logger.info(f"Starting new Gemini chat session: {session_id} (model: {name or 'account default'})")
                chat = self.client.start_chat(model=name)
                try:
                    response = await self._send(chat, message, images)
                    self.sessions[session_id] = _Session(chat, wanted, name)
                    return response
                except Exception as e:
                    if not is_fallback_error(e) or i == len(candidates) - 1:
                        logger.error(f"Error in session {session_id} get_response: {e}", exc_info=True)
                        raise
                    registry.mark_failed(name, e)
                    logger.warning(f"Retrying session {session_id} on '{candidates[i + 1]}'")

_translate_session_manager = None
_gemini_chat_manager = None

def init_session_managers():
    """
    Initialize session managers for translation and chat
    """
    global _translate_session_manager, _gemini_chat_manager
    try:
        client = get_gemini_client()
        _translate_session_manager = SessionManager(client)
        _gemini_chat_manager = SessionManager(client)
    except GeminiClientNotInitializedError:
        logger.warning("Session managers not initialized: Gemini client not available.")

def get_translate_session_manager():
    return _translate_session_manager

def get_gemini_chat_manager():
    return _gemini_chat_manager
