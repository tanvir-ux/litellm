"""Regression for #43014: Gemini chat streaming must map context-window 400s."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.exceptions import ContextWindowExceededError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.custom_httpx.http_handler import (
    MaskedHTTPStatusError,
    _raise_masked_async_error,
    _raise_masked_sync_error,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    make_call,
    make_sync_call,
)


GEMINI_CTX_BODY = (
    b'{"error": {"code": 400, "message": "The input token count (1200000) exceeds '
    b'the maximum number of tokens allowed (1048576).", "status": "INVALID_ARGUMENT"}}'
)


def _status_error(body: bytes = GEMINI_CTX_BODY) -> httpx.HTTPStatusError:
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
    )
    response = httpx.Response(400, request=request, content=body)
    return httpx.HTTPStatusError("Client error 400", request=request, response=response)


@pytest.mark.asyncio
async def test_masked_async_stream_error_message_is_str_not_bytes():
    with pytest.raises(MaskedHTTPStatusError) as exc_info:
        await _raise_masked_async_error(_status_error(), stream=True)

    err = exc_info.value
    assert isinstance(err.message, str)
    assert "exceeds the maximum number of tokens allowed" in err.message
    assert not err.message.startswith("b'")


def test_masked_sync_stream_error_message_is_str_not_bytes():
    with pytest.raises(MaskedHTTPStatusError) as exc_info:
        _raise_masked_sync_error(_status_error(), stream=True)

    err = exc_info.value
    assert isinstance(err.message, str)
    assert "exceeds the maximum number of tokens allowed" in err.message


@pytest.mark.asyncio
async def test_make_call_maps_masked_stream_context_window_error():
    """HTTP handler already consumed the body; make_call must still recover it."""
    with pytest.raises(MaskedHTTPStatusError) as masked_info:
        await _raise_masked_async_error(_status_error(), stream=True)
    masked = masked_info.value

    client = AsyncMock()
    client.post = AsyncMock(side_effect=masked)
    logging_obj = MagicMock()
    logging_obj.post_call = MagicMock()

    with pytest.raises(VertexAIError) as vertex_info:
        await make_call(
            client=client,
            gemini_client=None,
            api_base="https://example.com",
            headers={},
            data="{}",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            logging_obj=logging_obj,
        )

    vertex_err = vertex_info.value
    assert vertex_err.status_code == 400
    assert "exceeds the maximum number of tokens allowed" in vertex_err.message
    assert not vertex_err.message.startswith("b'")

    with pytest.raises(ContextWindowExceededError):
        exception_type(
            model="gemini-2.5-flash",
            custom_llm_provider="gemini",
            original_exception=vertex_err,
        )


def test_make_sync_call_maps_masked_stream_context_window_error():
    with pytest.raises(MaskedHTTPStatusError) as masked_info:
        _raise_masked_sync_error(_status_error(), stream=True)
    masked = masked_info.value

    client = MagicMock()
    client.post = MagicMock(side_effect=masked)
    logging_obj = MagicMock()
    logging_obj.post_call = MagicMock()

    with pytest.raises(VertexAIError) as vertex_info:
        make_sync_call(
            client=client,
            gemini_client=None,
            api_base="https://example.com",
            headers={},
            data="{}",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            logging_obj=logging_obj,
        )

    vertex_err = vertex_info.value
    assert "exceeds the maximum number of tokens allowed" in vertex_err.message

    with pytest.raises(ContextWindowExceededError):
        exception_type(
            model="gemini-2.5-flash",
            custom_llm_provider="vertex_ai_beta",
            original_exception=vertex_err,
        )
