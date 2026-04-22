# src/app/endpoints/google_generative.py
import hashlib
import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.logger import logger
from schemas.request import GoogleGenerativeRequest
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.session_manager import get_gemini_chat_manager

router = APIRouter()


def _build_tools_prompt(tools) -> str:
    """Convert Gemini functionDeclarations to a system prompt."""
    declarations = []
    for tool in tools:
        if tool.functionDeclarations:
            declarations.extend(tool.functionDeclarations)
    if not declarations:
        return ""
    lines = [
        "You have access to the following tools. When you want to call a tool, respond with "
        "ONLY a JSON object in this exact format, with no other text:\n"
        '{"functionCall": {"name": "<tool_name>", "args": {<arguments>}}}\n',
        "Available tools:",
    ]
    for decl in declarations:
        lines.append(f"- {decl.name}: {decl.description or ''}")
        if decl.parameters:
            lines.append(f"  Parameters: {json.dumps(decl.parameters)}")
    return "\n".join(lines)


def _parse_function_call(text: str) -> Optional[dict]:
    """Extract a functionCall JSON object from model response text."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and "functionCall" in obj:
                    return obj["functionCall"]
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _session_id_from_contents(contents) -> Optional[str]:
    """Derive a stable session ID from the first user message text."""
    for content in contents:
        if (content.role or "user") == "user":
            text = " ".join(p.text or "" for p in content.parts if p.text).strip()
            if text:
                return hashlib.md5(text.encode()).hexdigest()[:16]
    return None


def _last_user_message(contents) -> Optional[str]:
    """Extract the text of the last user turn."""
    for content in reversed(contents):
        if (content.role or "user") == "user":
            return " ".join(p.text or "" for p in content.parts if p.text).strip() or None
    return None


def _build_prompt_from_contents(contents) -> str:
    """Extract a flat text prompt from Gemini-style contents list."""
    parts_text = []
    for content in contents:
        role = content.role or "user"
        for part in content.parts:
            if part.text:
                parts_text.append(f"{role.capitalize()}: {part.text}")
            elif part.functionCall:
                fc = part.functionCall
                parts_text.append(
                    f"Assistant called tool {fc.get('name')}: {json.dumps(fc.get('args', {}))}"
                )
            elif part.functionResponse:
                fr = part.functionResponse
                resp = fr.get("response", {})
                parts_text.append(
                    f"Tool result for {fr.get('name')}: {json.dumps(resp)}"
                )
    return "\n\n".join(parts_text)


def _make_google_response(response_text: str, tools=None) -> dict:
    """Build the Google Generative AI response dict from model text."""
    # Check if the response is a function call
    if tools:
        fc = _parse_function_call(response_text)
        if fc:
            return {
                "candidates": [{
                    "content": {
                        "parts": [{"functionCall": fc}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
            }
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": response_text}],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
    }


@router.post("/v1beta/models/{model_path:path}")
async def google_generative_generate(model_path: str, request: GoogleGenerativeRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # model_path may be "gemini-2.5-flash:generateContent" or "gemini-2.5-flash:streamGenerateContent"
    parts = model_path.split(":")
    model_name = parts[0]
    action = parts[1] if len(parts) > 1 else "generateContent"
    is_streaming = action == "streamGenerateContent"

    try:
        session_id = _session_id_from_contents(request.contents)
        last_user_msg = _last_user_message(request.contents)

        if not last_user_msg:
            raise HTTPException(status_code=400, detail="No user message found in contents.")

        # Build any prefix (system instruction + tool defs) to prepend to the message
        prefix_parts = []
        if request.systemInstruction:
            if isinstance(request.systemInstruction, str):
                prefix_parts.append(f"System: {request.systemInstruction}")
            elif isinstance(request.systemInstruction, dict):
                for p in request.systemInstruction.get("parts", []):
                    if isinstance(p, dict) and p.get("text"):
                        prefix_parts.append(f"System: {p['text']}")
        if request.tools:
            tools_prompt = _build_tools_prompt(request.tools)
            if tools_prompt:
                prefix_parts.append(tools_prompt)

        # Use session manager for multi-turn persistence; session_id derived from first user message
        if session_id:
            session_manager = get_gemini_chat_manager()
            if not session_manager:
                raise HTTPException(status_code=503, detail="Session manager is not initialized.")
            prompt = "\n\n".join(prefix_parts + [last_user_msg]) if prefix_parts else last_user_msg
            logger.debug(f"Using session {session_id} for /v1beta request")
            response = await session_manager.get_response(model_name, prompt, session_id=session_id)
        else:
            # Fallback: stateless full-history prompt
            prompt_parts = prefix_parts + [_build_prompt_from_contents(request.contents)]
            prompt = "\n\n".join(p for p in prompt_parts if p)
            response = await gemini_client.generate_content(prompt, model_name)

        google_response = _make_google_response(response.text, request.tools)

        if is_streaming:
            async def sse_stream():
                yield f"data: {json.dumps(google_response)}\n\n"
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        return google_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /v1beta/models endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")
