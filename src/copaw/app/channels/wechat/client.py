# -*- coding: utf-8 -*-
"""HTTP client for Wechat protocol."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from .types import UploadResult

logger = logging.getLogger(__name__)


class WechatApiError(RuntimeError):
    """Raised when API returns non-success semantics."""


class WechatProtocolError(RuntimeError):
    """Raised when payload or protocol requirements are violated."""


class WechatApiClient:
    """Minimal async client for Wechat JSON APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        uin: str = "",
        timeout_ms: int = 15_000,
        long_poll_timeout_ms: int = 35_000,
        cdn_base_url: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._token = (token or "").strip()
        self._uin = (uin or "").strip()
        self._fallback_uin = str(secrets.randbelow(2**32))
        self._timeout_ms = max(1_000, timeout_ms)
        self._long_poll_timeout_ms = max(5_000, long_poll_timeout_ms)
        self._cdn_base_url = (cdn_base_url or "").strip().rstrip("/")
        if not self._uin:
            logger.warning(
                "wechat uin is empty; using instance-stable fallback identity",
            )
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_ms / 1000.0),
        )

    async def close(self) -> None:
        """Close underlying HTTP client."""
        await self._http.aclose()

    async def get_updates(self, get_updates_buf: str) -> Dict[str, Any]:
        """Long-poll updates from backend."""
        payload = {"get_updates_buf": get_updates_buf or ""}
        return await self._post_json(
            endpoint="ilink/bot/getupdates",
            payload=payload,
            timeout_ms=self._long_poll_timeout_ms,
            allow_timeout=True,
            allow_api_error=True,
        )

    async def send_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message payload."""
        return await self._post_json(
            endpoint="ilink/bot/sendmessage",
            payload=body,
        )

    async def get_upload_url(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Get upload params for CDN media upload."""
        return await self._post_json(
            endpoint="ilink/bot/getuploadurl",
            payload=body,
        )

    async def get_config(
        self,
        *,
        ilink_user_id: str,
        context_token: str = "",
    ) -> Dict[str, Any]:
        """Get config payload, including typing_ticket."""
        return await self._post_json(
            endpoint="ilink/bot/getconfig",
            payload={
                "ilink_user_id": ilink_user_id,
                "context_token": context_token,
            },
        )

    async def send_typing(
        self,
        *,
        ilink_user_id: str,
        typing_ticket: str,
        status: int,
    ) -> Dict[str, Any]:
        """Send typing state."""
        return await self._post_json(
            endpoint="ilink/bot/sendtyping",
            payload={
                "ilink_user_id": ilink_user_id,
                "typing_ticket": typing_ticket,
                "status": status,
            },
        )

    async def upload_media(
        self,
        *,
        local_path: Path,
        to_user_id: str,
        media_type: int,
    ) -> UploadResult:
        """Upload a local media file to CDN and return reference tokens."""
        if not local_path.is_file():
            raise WechatProtocolError(f"media file not found: {local_path}")
        raw = local_path.read_bytes()
        rawsize = len(raw)
        rawmd5 = hashlib.md5(raw).hexdigest()
        aes_key = secrets.token_bytes(16)
        ciphertext = _encrypt_aes_128_ecb(raw, aes_key)
        filekey = secrets.token_hex(16)

        upload_resp = await self.get_upload_url(
            {
                "filekey": filekey,
                "media_type": media_type,
                "to_user_id": to_user_id,
                "rawsize": rawsize,
                "rawfilemd5": rawmd5,
                "filesize": len(ciphertext),
                "no_need_thumb": True,
                "aeskey": aes_key.hex(),
            },
        )
        upload_param = str(upload_resp.get("upload_param") or "")
        if not upload_param:
            raise WechatProtocolError("getuploadurl returned empty upload_param")

        upload_url = self._build_cdn_upload_url(
            upload_param=upload_param,
            filekey=filekey,
        )
        response = await self._http.post(
            upload_url,
            content=ciphertext,
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code >= 400:
            raise WechatApiError(
                f"cdn upload failed: status={response.status_code}",
            )
        encrypted_query_param = response.headers.get(
            "x-encrypted-param",
            "",
        ).strip()
        if not encrypted_query_param:
            raise WechatProtocolError("cdn upload missing x-encrypted-param")

        return UploadResult(
            encrypted_query_param=encrypted_query_param,
            aes_key_b64=base64.b64encode(aes_key).decode("utf-8"),
            file_size=rawsize,
            file_size_cipher=len(ciphertext),
        )

    async def _post_json(
        self,
        *,
        endpoint: str,
        payload: Dict[str, Any],
        timeout_ms: Optional[int] = None,
        allow_timeout: bool = False,
        allow_api_error: bool = False,
    ) -> Dict[str, Any]:
        merged_payload = dict(payload)
        merged_payload["base_info"] = {"channel_version": "python-copaw-wechat"}
        body = httpx.Request(
            method="POST",
            url=self._base_url,
            json=merged_payload,
        ).read()
        headers = self._build_headers(content_length=len(body))
        timeout = timeout_ms if timeout_ms is not None else self._timeout_ms

        try:
            response = await self._http.post(
                endpoint,
                json=merged_payload,
                headers=headers,
                timeout=httpx.Timeout(timeout / 1000.0),
            )
        except httpx.TimeoutException:
            if allow_timeout:
                return {
                    "ret": 0,
                    "msgs": [],
                    "get_updates_buf": payload.get("get_updates_buf", ""),
                }
            raise

        if response.status_code >= 400:
            raise WechatApiError(
                f"{endpoint} http status={response.status_code} body={response.text[:300]}",
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise WechatProtocolError(
                f"{endpoint} returned invalid json",
            ) from exc
        if not isinstance(data, dict):
            raise WechatProtocolError(f"{endpoint} returned non-object json")
        ret = data.get("ret")
        if ret not in (None, 0) and not allow_api_error:
            errcode = data.get("errcode")
            errmsg = data.get("errmsg") or "unknown error"
            raise WechatApiError(
                f"{endpoint} ret={ret} errcode={errcode} errmsg={errmsg}",
            )
        return data

    def _build_headers(self, *, content_length: int) -> Dict[str, str]:
        uin_header = self._encoded_uin()
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Content-Length": str(content_length),
            "X-WECHAT-UIN": uin_header,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _encoded_uin(self) -> str:
        raw = self._uin or self._fallback_uin
        return base64.b64encode(raw.encode("utf-8")).decode("utf-8")

    def _build_cdn_upload_url(self, *, upload_param: str, filekey: str) -> str:
        if upload_param.startswith("http://") or upload_param.startswith("https://"):
            if "filekey=" in upload_param:
                return upload_param
            sep = "&" if "?" in upload_param else "?"
            return f"{upload_param}{sep}filekey={quote(filekey)}"

        cdn_base = self._cdn_base_url or self._base_url.rstrip("/")
        return (
            f"{cdn_base}/upload?encrypted_query_param={quote(upload_param)}"
            f"&filekey={quote(filekey)}"
        )


def _encrypt_aes_128_ecb(plain: bytes, key: bytes) -> bytes:
    """Encrypt bytes with AES-128-ECB and PKCS7 padding."""
    if len(key) != 16:
        raise WechatProtocolError("AES-128 key must be 16 bytes")
    try:
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import (
            Cipher,
            algorithms,
            modes,
        )
    except ImportError as exc:
        raise WechatProtocolError(
            "cryptography package is required for Wechat media upload",
        ) from exc

    padder = padding.PKCS7(128).padder()
    padded = padder.update(plain) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    return encryptor.update(padded) + encryptor.finalize()
