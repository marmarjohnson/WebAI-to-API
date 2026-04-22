import asyncio
from typing import Dict, Optional
from app.logger import logger
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError

class SessionManager:
    def __init__(self, client):
        self.client = client
        # Map session_id to Gemini chat session objects
        self.sessions: Dict[str, any] = {}
        self.lock = asyncio.Lock()

    async def get_response(self, model: str, message: str, session_id: str = "default", images=None):
        async with self.lock:
            # If the session doesn't exist, start a new chat
            if session_id not in self.sessions:
                logger.info(f"Starting new Gemini chat session: {session_id}")
                # If model is an Enum, use its value
                model_value = model.value if hasattr(model, "value") else model
                self.sessions[session_id] = self.client.start_chat(model=model_value)

            session = self.sessions[session_id]
            
            try:
                # send_message uses the underlying WebAI persistence
                return await session.send_message(prompt=message, files=images)
            except Exception as e:
                logger.error(f"Error in session {session_id} get_response: {e}", exc_info=True)
                # If the session is expired/invalid, we remove it so it can be recreated
                del self.sessions[session_id]
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
