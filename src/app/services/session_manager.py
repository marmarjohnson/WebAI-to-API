import asyncio
from typing import Dict, Optional, Tuple
from app.logger import logger
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError

class SessionManager:
    def __init__(self, client):
        self.client = client
        # Map session_id to (chat_session, model_name)
        self.sessions: Dict[str, Tuple[any, str]] = {}
        self.lock = asyncio.Lock()

    async def get_response(self, model: str, message: str, session_id: str = "default", images=None):
        async with self.lock:
            model_value = model.value if hasattr(model, "value") else model
            existing = self.sessions.get(session_id)

            if existing is None:
                logger.info(f"Starting new Gemini chat session: {session_id} (model: {model_value})")
                chat = self.client.start_chat(model=model_value)
                self.sessions[session_id] = (chat, model_value)
            elif existing[1] != model_value:
                logger.info(f"Model changed for session {session_id}: {existing[1]} → {model_value}, restarting session")
                chat = self.client.start_chat(model=model_value)
                self.sessions[session_id] = (chat, model_value)
            else:
                chat = existing[0]

            session = chat
            
            try:
                # send_message uses the underlying WebAI persistence
                return await session.send_message(prompt=message, files=images)
            except Exception as e:
                logger.error(f"Error in session {session_id} get_response: {e}", exc_info=True)
                self.sessions.pop(session_id, None)
                raise

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
