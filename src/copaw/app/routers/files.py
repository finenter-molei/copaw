# -*- coding: utf-8 -*-
"""聊天附件（图片、文档等）的上传与访问 API。"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from ...constant import (
    WORKING_DIR,
    MEDIA_MAX_AGE_DAYS,
    MEDIA_MAX_SIZE_MB,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])

MEDIA_DIR = WORKING_DIR / "media"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


def _ensure_media_dir() -> Path:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR


def run_media_cleanup() -> tuple[int, int]:
    """
    从 WORKING_DIR/media 删除过期/超量文件，避免磁盘无限增长。

    使用 COPAW_MEDIA_MAX_AGE_DAYS（超过 N 天删除）与 COPAW_MEDIA_MAX_SIZE_MB
    （总大小上限，按最旧优先删）。返回 (删除数量, 释放字节数)。
    """
    if not MEDIA_DIR.is_dir():
        return 0, 0
    now = time.time()
    max_age_sec = (MEDIA_MAX_AGE_DAYS * 86400) if MEDIA_MAX_AGE_DAYS > 0 else 0
    max_size_bytes = (MEDIA_MAX_SIZE_MB * 1024 * 1024) if MEDIA_MAX_SIZE_MB > 0 else 0

    entries: list[tuple[Path, float, int]] = []
    for p in MEDIA_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            stat = p.stat()
            entries.append((p, stat.st_mtime, stat.st_size))
        except OSError:
            continue

    deleted = 0
    freed = 0

    # 按天数删除
    if max_age_sec > 0:
        cutoff = now - max_age_sec
        for p, mtime, size in entries:
            if mtime < cutoff:
                try:
                    p.unlink()
                    deleted += 1
                    freed += size
                except OSError as e:
                    logger.warning("Media cleanup: failed to remove %s: %s", p, e)
        entries = [(p, m, s) for p, m, s in entries if p.exists()]

    # 按总大小删除（最旧优先）
    if max_size_bytes > 0 and entries:
        total = sum(s for _, _, s in entries)
        if total > max_size_bytes:
            entries.sort(key=lambda x: x[1])
            for p, _, size in entries:
                if total <= max_size_bytes:
                    break
                try:
                    p.unlink()
                    deleted += 1
                    freed += size
                    total -= size
                except OSError as e:
                    logger.warning("Media cleanup: failed to remove %s: %s", p, e)

    if deleted:
        logger.info(
            "Media cleanup: removed %s file(s), freed %s bytes",
            deleted,
            freed,
        )
    return deleted, freed


def _safe_filename(original: str) -> str:
    """生成唯一文件名并保留原扩展名。"""
    suffix = Path(original).suffix if original else ""
    short_hash = hashlib.md5(
        f"{uuid.uuid4()}{original}".encode()
    ).hexdigest()[:12]
    stem = Path(original).stem if original else "file"
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    safe_stem = safe_stem[:60]
    return f"{safe_stem}_{short_hash}{suffix}"


@router.post(
    "/upload",
    summary="上传文件（图片、文档等）",
    description="上传单个文件作为聊天附件，返回可在消息中引用的 URL。",
)
async def upload_file(
    file: UploadFile = File(
        ...,
        description="要上传的文件",
    ),
) -> dict:
    """将上传文件保存到 WORKING_DIR/media 并返回其访问 URL。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    media_dir = _ensure_media_dir()
    safe_name = _safe_filename(file.filename)
    dest = media_dir / safe_name

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    dest.write_bytes(data)

    content_type = (
        file.content_type
        or mimetypes.guess_type(safe_name)[0]
        or "application/octet-stream"
    )

    return {
        "url": f"/files/serve/{safe_name}",
        "filename": file.filename,
        "size": len(data),
        "content_type": content_type,
    }


@router.get(
    "/serve/{filename:path}",
    summary="访问已上传文件",
    description="从媒体目录流式返回已上传的文件。",
)
async def serve_file(filename: str):
    """从 WORKING_DIR/media 返回指定文件。"""
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = MEDIA_DIR / filename
    resolved = file_path.resolve()
    if not str(resolved).startswith(str(MEDIA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(resolved, media_type=media_type)
