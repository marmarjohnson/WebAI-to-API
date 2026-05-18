# src/app/endpoints/chat.py
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.logger import logger
from schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.session_manager import get_translate_session_manager, get_gemini_chat_manager

router = APIRouter()

@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        response = await session_manager.get_response(request.model, request.message, images=request.files or None)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


def _build_tools_prompt(tools: list) -> str:
    """Convert OpenAI tool definitions to a system prompt for Gemini."""
    declarations = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            declarations.append(t["function"])
    if not declarations:
        return ""
    lines = [
        "You have access to the following tools. When you want to call a tool, respond with "
        "ONLY a JSON object in this exact format, with no other text before or after:\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<arguments>}}}\n',
        "Available tools:",
    ]
    for fn in declarations:
        lines.append(f"- {fn['name']}: {fn.get('description', '')}")
        if fn.get("parameters"):
            lines.append(f"  Parameters: {json.dumps(fn['parameters'])}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> Optional[dict]:
    """Extract a tool_call JSON object from model response text."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and "tool_call" in obj:
                    return obj["tool_call"]
            except (json.JSONDecodeError, ValueError):
                pass
    return None


async def _sse_chat_generator(chat_id: str, model: str, response_text: str, tool_call: Optional[dict]):
    ts = int(time.time())
    if tool_call:
        args = tool_call.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        call_id = f"call_{chat_id[-8:]}"
        name = tool_call.get("name", "")
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {
                     "tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                     "function": {"name": name, "arguments": args_str}}]
                 }, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    else:
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": ts, "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def convert_to_openai_format(response_text: str, model: str, stream: bool = False, tool_call: Optional[dict] = None):
    ts = int(time.time())
    if tool_call:
        args = tool_call.get("arguments", {})
        return {
            "id": f"chatcmpl-{ts}",
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "created": ts,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{ts}",
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(args) if isinstance(args, dict) else args,
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return {
        "id": f"chatcmpl-{ts}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": ts,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@router.get("/v1/models")
async def list_models():
    from gemini_webapi.constants import Model
    ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_name,
                "object": "model",
                "created": ts,
                "owned_by": "google",
            }
            for model in Model
            if model != Model.UNSPECIFIED
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    is_stream = request.stream if request.stream is not None else False

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    if not request.model:
        raise HTTPException(status_code=400, detail="Model not specified in the request.")

    # Session persistence path: when a user/session ID is provided, use the session manager
    # so Gemini retains conversation history on its side. Only the latest user message is sent.
    if request.user:
        session_manager = get_gemini_chat_manager()
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager is not initialized.")

        # Extract the last user message — that's the new turn; history lives in the session
        last_user_content = next(
            (msg.get("content", "") for msg in reversed(request.messages) if msg.get("role") == "user"),
            None,
        )
        if not last_user_content:
            raise HTTPException(status_code=400, detail="No user message found.")

        prompt = last_user_content
        if request.tools:
            tools_prompt = _build_tools_prompt(request.tools)
            if tools_prompt:
                prompt = f"{tools_prompt}\n\n{prompt}"

        try:
            response = await session_manager.get_response(request.model, prompt, session_id=request.user)
            logger.debug(f"Gemini session response: {response.text!r}")
            tool_call = _parse_tool_call(response.text) if request.tools else None
            logger.debug(f"Parsed tool_call: {tool_call}")
            if is_stream:
                chat_id = f"chatcmpl-{int(time.time())}"
                return StreamingResponse(_sse_chat_generator(chat_id, request.model, response.text, tool_call), media_type="text/event-stream")
            return convert_to_openai_format(response.text, request.model, is_stream, tool_call)
        except Exception as e:
            logger.error(f"Error in /v1/chat/completions session endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")

    # Stateless path: use fingerprint-based session reuse so a single Gemini WebUI
    # session handles all turns of the same conversation (avoids spawning a new session per turn).
    session_manager = get_gemini_chat_manager()
    msg_count = len(request.messages)
    key = _session_key(request.messages)
    tools_prompt = _build_tools_prompt(request.tools) if request.tools else ""

    existing_chat: Optional[_ConversationSession] = None
    if key and session_manager:
        candidate = _chat_sessions.get(key)
        if candidate and msg_count > candidate.last_input_count:
            existing_chat = candidate

    try:
        response_text: str = ""
        if existing_chat:
            # Continuation: send only the new user message — Gemini session holds history.
            new_messages = request.messages[existing_chat.last_input_count:]
            last_user = next(
                (msg.get("content", "") for msg in reversed(new_messages) if msg.get("role") == "user"),
                None,
            )
            prompt = last_user or ""
            if tools_prompt and prompt:
                prompt = f"{tools_prompt}\n\n{prompt}"
            if prompt:
                logger.info(
                    f"chat_completions: continuing session {existing_chat.gemini_session_id[:8]} "
                    f"({existing_chat.last_input_count}→{msg_count} msgs)"
                )
                response = await session_manager.get_response(request.model, prompt, session_id=existing_chat.gemini_session_id)
                response_text = response.text
                existing_chat.last_input_count = msg_count

        if not response_text:
            # Fresh start: build full prompt and open a new Gemini session.
            conversation_parts = []
            if tools_prompt:
                conversation_parts.append(tools_prompt)
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content") or ""
                if role == "system":
                    conversation_parts.append(f"System: {content}")
                elif role == "user":
                    conversation_parts.append(f"User: {content}")
                elif role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            conversation_parts.append(
                                f"Assistant called tool {fn.get('name')}: {fn.get('arguments', '')}"
                            )
                    elif content:
                        conversation_parts.append(f"Assistant: {content}")
                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    conversation_parts.append(f"Tool result [{tool_call_id}]: {content}")

            if not conversation_parts:
                raise HTTPException(status_code=400, detail="No valid messages found.")

            final_prompt = "\n\n".join(conversation_parts)
            session_id = uuid.uuid4().hex

            if session_manager and key:
                logger.info(f"chat_completions: starting new session {session_id[:8]} (key={key})")
                response = await session_manager.get_response(request.model, final_prompt, session_id=session_id)
                response_text = response.text
                _chat_sessions[key] = _ConversationSession(gemini_session_id=session_id, last_input_count=msg_count)
            else:
                response = await gemini_client.generate_content(message=final_prompt, model=request.model, files=None)
                response_text = response.text

        logger.debug(f"Gemini raw response: {response_text!r}")
        tool_call = _parse_tool_call(response_text) if request.tools else None
        logger.debug(f"Parsed tool_call: {tool_call}")
        if is_stream:
            chat_id = f"chatcmpl-{int(time.time())}"
            return StreamingResponse(_sse_chat_generator(chat_id, request.model, response_text, tool_call), media_type="text/event-stream")
        return convert_to_openai_format(response_text, request.model, is_stream, tool_call)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")


# ---------------------------------------------------------------------------
# OpenAI Responses API  (used by opencode and other newer OpenAI SDK clients)
# ---------------------------------------------------------------------------

class OpenAIResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: List[dict] = []
    tools: Optional[List[dict]] = None
    stream: Optional[bool] = False
    previous_response_id: Optional[str] = None


@dataclass
class _ConversationSession:
    gemini_session_id: str
    last_input_count: int


# Keyed by a hash of the conversation's system prompt.
# Allows turns from the same opencode session to reuse one Gemini chat even
# when the client never sends previous_response_id.
_conversation_sessions: Dict[str, _ConversationSession] = {}

# Same fingerprint scheme for /v1/chat/completions clients (e.g. pi).
_chat_sessions: Dict[str, _ConversationSession] = {}

# Maps response_id → session key, for clients that do send previous_response_id.
_responses_session_map: Dict[str, str] = {}


def _session_key(input_messages: list) -> Optional[str]:
    """Return a stable fingerprint for the conversation, derived from its system prompt."""
    for msg in input_messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if c.get("text"))
            if content:
                return hashlib.sha256(content[:400].encode()).hexdigest()[:20]
    # No system message: fall back to first user message
    for msg in input_messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if c.get("text"))
            if content:
                return hashlib.sha256(content[:200].encode()).hexdigest()[:20]
    return None


_SYSTEM_MSG_MAX = 2000   # chars kept from opencode's large system prompt
_TOOL_DESC_MAX  = 120    # chars kept per tool description

def _build_responses_tools_prompt(tools: list) -> str:
    """Convert Responses API flat tool definitions to a compact system prompt.
    Intentionally omits full parameter schemas — the Gemini web interface cannot
    handle the 36KB+ payload opencode sends verbatim."""
    lines = [
        "You have access to the following tools. When you want to call a tool, respond with "
        "ONLY a JSON object in this exact format, with no other text before or after:\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<arguments>}}}\n',
        "Available tools:",
    ]
    found = False
    for t in tools:
        if t.get("type") == "function":
            found = True
            name = t.get("name", "")
            desc = (t.get("description", "") or "")[:_TOOL_DESC_MAX]
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if found else ""


def _responses_input_to_prompt(input_messages: list) -> str:
    """Convert Responses API input array to flat text prompt."""
    parts = []
    for msg in input_messages:
        msg_type = msg.get("type")
        if msg_type == "function_call_output":
            parts.append(f"Tool result [{msg.get('call_id', '')}]: {msg.get('output', '')}")
            continue
        if msg_type == "function_call":
            parts.append(f"Assistant called tool {msg.get('name', '')}: {msg.get('arguments', '{}')}")
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if c.get("type") in ("input_text", "text", "output_text") and c.get("text")
            )
        # Truncate huge system prompts (opencode sends 16KB+) to stay within
        # Gemini web interface limits
        if role == "system" and len(content) > _SYSTEM_MSG_MAX:
            content = content[:_SYSTEM_MSG_MAX] + "\n[truncated]"
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, "User")
        if content:
            parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


def _responses_new_turn_prompt(input_messages: list, tools_prompt: str) -> str:
    """Extract only the new messages from a Responses API continuation request.

    When previous_response_id is set, the client sends only new content (tool
    results + latest user message).  We skip system messages here because the
    Gemini session already has that context from the first turn.
    """
    parts = []
    if tools_prompt:
        parts.append(tools_prompt)
    for msg in input_messages:
        msg_type = msg.get("type")
        if msg_type == "function_call_output":
            parts.append(f"Tool result [{msg.get('call_id', '')}]: {msg.get('output', '')}")
            continue
        if msg_type == "function_call":
            parts.append(f"Assistant called tool {msg.get('name', '')}: {msg.get('arguments', '{}')}")
            continue
        role = msg.get("role", "")
        if role == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if c.get("type") in ("input_text", "text", "output_text") and c.get("text")
            )
        prefix = {"user": "User", "assistant": "Assistant"}.get(role, "User")
        if content:
            parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


def _build_responses_json(resp_id, msg_id, model, text, tool_call, created_at) -> dict:
    if tool_call:
        args_str = json.dumps(tool_call.get("arguments", {})) if isinstance(tool_call.get("arguments"), dict) else str(tool_call.get("arguments", "{}"))
        output = [{
            "type": "function_call",
            "id": f"fc_{resp_id[-8:]}",
            "call_id": f"call_{resp_id[-8:]}",
            "name": tool_call.get("name", ""),
            "arguments": args_str,
            "status": "completed",
        }]
    else:
        output = [{
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
            "status": "completed",
        }]
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


async def _responses_sse_generator(resp_id, msg_id, model, text, tool_call, created_at):
    base = {"id": resp_id, "object": "response", "created_at": created_at,
            "status": "in_progress", "model": model, "output": []}
    yield f"data: {json.dumps({'type': 'response.created', 'response': base})}\n\n"

    if tool_call:
        fc_id = f"fc_{resp_id[-8:]}"
        call_id = f"call_{resp_id[-8:]}"
        args_str = json.dumps(tool_call.get("arguments", {})) if isinstance(tool_call.get("arguments"), dict) else str(tool_call.get("arguments", "{}"))
        name = tool_call.get("name", "")
        item = {"id": fc_id, "type": "function_call", "name": name,
                "arguments": "", "call_id": call_id, "status": "in_progress"}
        yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': item})}\n\n"
        yield f"data: {json.dumps({'type': 'response.function_call_arguments.delta', 'item_id': fc_id, 'output_index': 0, 'delta': args_str})}\n\n"
        yield f"data: {json.dumps({'type': 'response.function_call_arguments.done', 'item_id': fc_id, 'output_index': 0, 'arguments': args_str})}\n\n"
        item = {**item, "arguments": args_str, "status": "completed"}
        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': item})}\n\n"
        completed = {**base, "status": "completed", "output": [item],
                     "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        yield f"data: {json.dumps({'type': 'response.completed', 'response': completed})}\n\n"
    else:
        item = {"id": msg_id, "type": "message", "role": "assistant",
                "content": [], "status": "in_progress"}
        yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': item})}\n\n"
        yield f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': ''}})}\n\n"
        yield f"data: {json.dumps({'type': 'response.output_text.delta', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'delta': text})}\n\n"
        yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': text})}\n\n"
        item = {**item, "content": [{"type": "output_text", "text": text}], "status": "completed"}
        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': item})}\n\n"
        completed = {**base, "status": "completed", "output": [item],
                     "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        yield f"data: {json.dumps({'type': 'response.completed', 'response': completed})}\n\n"


@router.post("/v1/responses")
async def responses_api(request: OpenAIResponsesRequest):
    try:
        gemini_client = get_gemini_client()  # raises if not ready; also used as fallback below
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    logger.debug(
        f"Responses API: previous_response_id={request.previous_response_id!r} "
        f"input_count={len(request.input)} model={request.model}"
    )

    if not request.input:
        raise HTTPException(status_code=400, detail="No input messages provided.")
    if not request.model:
        raise HTTPException(status_code=400, detail="Model not specified.")

    tools_prompt = _build_responses_tools_prompt(request.tools) if request.tools else ""

    session_manager = get_gemini_chat_manager()

    try:
        response_text: str = ""
        gemini_session_id: str = ""
        new_input_count = len(request.input)
        key = _session_key(request.input)

        # --- Determine which Gemini session to use ---
        # Priority 1: explicit previous_response_id from the client (future-proof)
        explicit_session_key = (
            _responses_session_map.get(request.previous_response_id)
            if request.previous_response_id else None
        )

        existing: Optional[_ConversationSession] = None
        if explicit_session_key:
            existing = _conversation_sessions.get(explicit_session_key)
            key = explicit_session_key

        # Priority 2: fingerprint-based detection — same system prompt AND growing history
        if existing is None and key:
            candidate = _conversation_sessions.get(key)
            if candidate and new_input_count > candidate.last_input_count:
                existing = candidate

        if existing and session_manager:
            # Continuation: send only the new messages appended since last turn.
            new_messages = request.input[existing.last_input_count:]
            prompt = _responses_new_turn_prompt(new_messages, tools_prompt)
            if not prompt:
                # Nothing new to say — fall through and treat as fresh to avoid a hang.
                existing = None
            else:
                gemini_session_id = existing.gemini_session_id
                logger.info(
                    f"Responses API: continuing session {gemini_session_id[:8]} "
                    f"({existing.last_input_count}→{new_input_count} msgs)"
                )
                response = await session_manager.get_response(request.model, prompt, session_id=gemini_session_id)
                response_text = response.text
                existing.last_input_count = new_input_count

        if not response_text:
            # New session: send full context.
            conversation_parts: List[str] = []
            if tools_prompt:
                conversation_parts.append(tools_prompt)
            prompt_text = _responses_input_to_prompt(request.input)
            if prompt_text:
                conversation_parts.append(prompt_text)
            if not conversation_parts:
                raise HTTPException(status_code=400, detail="No valid messages found.")
            final_prompt = "\n\n".join(conversation_parts)
            gemini_session_id = uuid.uuid4().hex

            if session_manager:
                logger.info(f"Responses API: starting new session {gemini_session_id[:8]} (key={key})")
                response = await session_manager.get_response(
                    request.model, final_prompt, session_id=gemini_session_id
                )
                response_text = response.text
                if key:
                    _conversation_sessions[key] = _ConversationSession(
                        gemini_session_id=gemini_session_id,
                        last_input_count=new_input_count,
                    )
            else:
                response = await gemini_client.generate_content(message=final_prompt, model=request.model, files=None)
                response_text = response.text

        tool_call = _parse_tool_call(response_text) if request.tools else None
        logger.debug(f"Responses API: model={request.model} tool_call={tool_call is not None}")

        resp_id = f"resp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        created_at = int(time.time())

        # Link this response to its session so previous_response_id works for other clients.
        if key and session_manager:
            _responses_session_map[resp_id] = key

        if request.stream:
            return StreamingResponse(
                _responses_sse_generator(resp_id, msg_id, request.model, response_text, tool_call, created_at),
                media_type="text/event-stream",
            )
        return _build_responses_json(resp_id, msg_id, request.model, response_text, tool_call, created_at)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /v1/responses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
