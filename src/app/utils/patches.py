"""
Runtime monkey-patches for third-party library bugs.
Call apply_all() once at process startup before any library code runs.
"""
import logging

logger = logging.getLogger(__name__)


def apply_all() -> None:
    # gemini-webapi >= 2.0 uses curl_cffi instead of httpx, so the httpx
    # CookieConflict patch below no longer applies. Kept for reference only.
    pass


def _patch_httpx_cookie_conflict() -> None:
    """
    httpx.Cookies raises CookieConflict when the same cookie name exists for
    multiple domains (e.g. __Secure-1PSID on both .google.com and
    gemini.google.com). gemini_webapi triggers this on every re-init after the
    first request, because Google responses accumulate the cookie across domains.

    Fix: patch __contains__ and __getitem__ to fall back to _jar iteration so
    they return the first matching value instead of raising.
    """
    try:
        from httpx._models import Cookies

        _orig_contains = Cookies.__contains__
        _orig_getitem = Cookies.__getitem__

        def _safe_contains(self, name: str) -> bool:
            try:
                return _orig_contains(self, name)
            except Exception:
                if hasattr(self, "_jar"):
                    return any(item[2] == name for item in self._jar)
                return False

        def _safe_getitem(self, name: str) -> str:
            try:
                return _orig_getitem(self, name)
            except Exception:
                if hasattr(self, "_jar"):
                    for item in self._jar:
                        if item[2] == name:
                            return item[3]
                raise

        Cookies.__contains__ = _safe_contains
        Cookies.__getitem__ = _safe_getitem
        logger.debug("Applied httpx.Cookies patch for CookieConflict (gemini_webapi multi-domain)")
    except Exception as exc:
        logger.warning(f"Could not apply httpx.Cookies CookieConflict patch: {exc}")
