# src/app/services/model_registry.py
"""
Model availability tracking and graceful downgrade.

Gemini WebUI access can change underneath us (e.g. a Pro/Plus subscription
lapsing). Rather than let requests for a model the account can no longer use
hang or fail, every request is resolved through this registry:

1. The requested name is normalized to a model family the account offers
   (gemini-webapi aliases first, then a family heuristic, so names like
   "gemini-2.5-pro" or "gemini-3.0-flash-thinking" still work).
2. A downgrade chain is built (Pro -> Flash -> Flash-Lite), filtered by the
   account's live model list (GET_USER_STATUS, re-fetched every few minutes,
   which reports per-model `is_available`) and by models that recently failed.
3. The caller tries each candidate in order; models that fail with
   availability-type errors are put on cooldown so later requests skip them.

The chain only ever steps *down*; a Flash request is never upgraded to Pro.
"""
import time
from typing import List, Optional, Set, Tuple

from gemini_webapi.exceptions import (
    APIError,
    GeminiError,
    ModelInvalidError,
    TemporarilyBlockedError,
    UsageLimitExceededError,
)

from app.config import CONFIG
from app.logger import logger

PRO, FLASH, FLASH_LITE = "gemini-pro", "gemini-flash", "gemini-flash-lite"

# Best first. A request enters the chain at its own tier and only moves down.
PREFERENCE: List[str] = [PRO, FLASH, FLASH_LITE]

# Empty string = no model header: Google picks the account's default model.
DEFAULT_MODEL = ""

LIVE_LIST_TTL = 10 * 60          # seconds between account/model refreshes
DEGRADED_RECHECK = 60            # re-check sooner while the account status is not AVAILABLE
HARD_FAILURE_COOLDOWN = 30 * 60  # model rejected / usage limit
SOFT_FAILURE_COOLDOWN = 5 * 60   # stalls, timeouts, other API errors


def request_timeout() -> float:
    """Upper bound for one model attempt, so a stalled stream falls back instead of hanging."""
    try:
        return CONFIG.getfloat("AI", "request_timeout", fallback=180.0)
    except ValueError:
        return 180.0


def is_fallback_error(exc: BaseException) -> bool:
    """Errors worth retrying on a lower-tier model. IP blocks are not model-specific."""
    if isinstance(exc, TemporarilyBlockedError):
        return False
    return isinstance(exc, (ModelInvalidError, UsageLimitExceededError, APIError, GeminiError, TimeoutError))


def _family_of(name: str) -> str:
    lowered = name.lower()
    if "pro" in lowered:
        return PRO
    if "lite" in lowered:
        return FLASH_LITE
    return FLASH  # flash, flash-thinking, anything unrecognized


def canonical_model(web_client, name) -> str:
    """Map any requested model name onto a model family (or DEFAULT_MODEL)."""
    name = (getattr(name, "value", name) or "").strip()
    if name.lower() in ("", "unspecified", "default", "auto"):
        return DEFAULT_MODEL
    try:
        resolved = web_client.resolve_model(name).model_name
        if resolved in PREFERENCE:
            return resolved
        return _family_of(resolved)
    except ValueError:
        pass
    mapped = _family_of(name)
    if mapped != name:
        logger.info(f"Model '{name}' is not a Gemini WebUI model name; using '{mapped}'")
    return mapped


class ModelRegistry:
    def __init__(self) -> None:
        self._live: Optional[Set[str]] = None
        self._display: dict = {}
        self._tier: Optional[str] = None
        self._live_fetched_at: float = 0.0
        self._cooldown_until: dict = {}

    # ---- live list ---------------------------------------------------------

    async def refresh(self, web_client, force: bool = False) -> None:
        """Re-read which models this account can use (and its tier) from Gemini."""
        if not force and time.time() - self._live_fetched_at < LIVE_LIST_TTL:
            return
        self._live_fetched_at = time.time()

        # gemini-webapi only ever *adds* to its registry, so a model the account loses
        # would linger. Rebuild from scratch; keep the old one if the fetch fails.
        previous = dict(web_client._model_registry)
        try:
            web_client._model_registry = {}
            await web_client._fetch_user_status()
        except Exception as e:
            logger.warning(f"Gemini model list refresh failed: {e}")
        if not web_client._model_registry:
            web_client._model_registry = previous
            if self._live is None and not previous:
                return

        status = getattr(web_client, "account_status", None)
        if status is not None and getattr(status, "name", "AVAILABLE") != "AVAILABLE":
            # e.g. a transient UNAUTHENTICATED (guest) session hides every model but the
            # default. Honor it (those requests would fail anyway) but re-check soon.
            logger.warning(
                f"Gemini account status {status.name}: {getattr(status, 'description', '')}; "
                f"re-checking in {DEGRADED_RECHECK}s"
            )
            self._live_fetched_at = time.time() - LIVE_LIST_TTL + DEGRADED_RECHECK

        models = list(web_client._model_registry.values())
        new_live = {m.model_name for m in models if m.is_available}
        self._display = {m.model_name: m.display_name for m in models}
        if self._live is not None and new_live != self._live:
            lost, gained = self._live - new_live, new_live - self._live
            if lost:
                logger.warning(f"Gemini account lost access to: {sorted(lost)}. Requests will downgrade.")
            if gained:
                logger.info(f"Gemini account gained access to: {sorted(gained)}")
        elif self._live is None:
            logger.info(f"Gemini models available: {self.describe(new_live)}")
        self._live = new_live

        try:
            await web_client._fetch_usage_info()
            tier = (web_client.usage_info or {}).get("tier", {}).get("label")
            if tier and self._tier and tier != self._tier:
                logger.warning(f"Gemini account tier changed: {self._tier} -> {tier}")
            elif tier and not self._tier:
                logger.info(f"Gemini account tier: {tier}")
            self._tier = tier or self._tier
        except Exception as e:
            logger.debug(f"Gemini usage info refresh failed: {e}")

    def describe(self, names) -> List[str]:
        return [f"{n} ({self._display.get(n, '?')})" for n in PREFERENCE if n in names]

    @property
    def live_models(self) -> Optional[List[str]]:
        if self._live is None:
            return None
        return [m for m in PREFERENCE if m in self._live] + sorted(self._live - set(PREFERENCE))

    @property
    def tier(self) -> Optional[str]:
        return self._tier

    # ---- cooldowns ---------------------------------------------------------

    def mark_failed(self, model: str, exc: BaseException) -> None:
        # "<model> is not available for use" = the account lacks the entitlement, not a hiccup.
        hard = isinstance(exc, (ModelInvalidError, UsageLimitExceededError)) or (
            isinstance(exc, GeminiError) and "not available" in str(exc)
        )
        seconds = HARD_FAILURE_COOLDOWN if hard else SOFT_FAILURE_COOLDOWN
        self._cooldown_until[model] = time.time() + seconds
        logger.warning(f"Model '{model}' failed ({type(exc).__name__}); skipping it for {seconds // 60} min")
        # The account's entitlements may have changed: re-check on next request.
        self._live_fetched_at = 0.0

    def _cooling(self, model: str) -> bool:
        until = self._cooldown_until.get(model)
        if until and until > time.time():
            return True
        self._cooldown_until.pop(model, None)
        return False

    # ---- resolution --------------------------------------------------------

    async def candidates(self, web_client, requested) -> List[str]:
        """Ordered list of models to try for a request (never empty)."""
        return (await self.resolve(web_client, requested))[1]

    async def resolve(self, web_client, requested) -> Tuple[str, List[str]]:
        """(canonical requested model, ordered models to try)."""
        await self.refresh(web_client)
        wanted = canonical_model(web_client, requested)
        if wanted == DEFAULT_MODEL:
            return wanted, [DEFAULT_MODEL]

        chain = PREFERENCE[PREFERENCE.index(wanted):] if wanted in PREFERENCE else [wanted]
        usable = [m for m in chain if (self._live is None or m in self._live) and not self._cooling(m)]
        if not usable:
            # Everything filtered out: still try something rather than fail outright,
            # preferring models the account lists, else let Google pick its default.
            listed = [m for m in chain if self._live is None or m in self._live]
            usable = listed or [DEFAULT_MODEL]
        if usable[0] != wanted:
            logger.warning(f"Model '{wanted}' unavailable; downgrading to '{usable[0] or 'account default'}'")
        return wanted, usable


registry = ModelRegistry()
