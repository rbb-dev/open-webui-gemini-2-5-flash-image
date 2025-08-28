"""
title: gemini-2.5-Flash-Image
description: description: Generate or edit images using Gemini 2.5 Flash via the official Google generateContent API
id: gemini_flash_img
author: rbb-dev
author_url: https://github.com/rbb-dev/
version: 0.1.5
"""

# if no image uploaded with user turn, will look for the last image in the chat and send it to the model with the prompt (editign image)


import base64
import io
import json
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal
import httpx
from fastapi import Request, UploadFile, BackgroundTasks
from open_webui.routers.files import upload_file
from open_webui.models.users import UserModel, Users
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.datastructures import Headers


class Pipe:
    class Valves(BaseModel):
        COMET_API_KEY: str = Field(default="", description="Your API key")
        API_BASE_URL: str = Field(
            default="https://api.CometAPI.com/v1beta", description="Gemini API base URL"
        )
        REQUEST_TIMEOUT: int = Field(
            default=600, description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipes(self) -> List[dict]:
        return [{"id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        user = Users.get_user_by_id(__user__["id"])

        async def _send_status(msg: str) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": msg, "done": False}}
                )

        async def stream_response():
            try:
                model = body.get("model", "gemini-2.5-flash-image")
                messages = body.get("messages", [])
                is_stream = body.get("stream", False)

                if not self.valves.COMET_API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: Comet API key not provided.",
                    )
                    return

                last_user_msg = next(
                    (msg for msg in reversed(messages) if msg.get("role") == "user"),
                    None,
                )
                if not last_user_msg:
                    yield self._format_data(
                        is_stream=is_stream, content="Error: No user message found."
                    )
                    return

                content = last_user_msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ]
                    prompt = " ".join(text_parts)
                else:
                    prompt = content
                if not prompt.strip():
                    yield self._format_data(
                        is_stream=is_stream, content="Error: No prompt provided."
                    )
                    return

                await _send_status("Processing request…")
                image_base64 = await self._find_image(messages)

                if not image_base64 and len(messages) == 1:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Please provide an image for editing or a description for generation.",
                    )
                    return

                if image_base64:
                    mime_type = (
                        re.search(r"data:([^;]+);base64,", image_base64).group(1)
                        if image_base64.startswith("data:")
                        else "image/png"
                    )
                    raw_base64 = image_base64.split(";base64,")[-1]
                    payload = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": raw_base64,
                                        }
                                    },
                                ],
                            }
                        ],
                        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                    }
                    await _send_status("Sending image for editing…")
                else:
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    await _send_status("Sending generation request…")

                await _send_status("Awaiting response…")
                full_url = f"{self.valves.API_BASE_URL}/models/gemini-2.5-flash-image-preview:generateContent"
                headers = {
                    "Authorization": self.valves.COMET_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "Connection": "keep-alive",
                }

                async with httpx.AsyncClient(
                    base_url=self.valves.API_BASE_URL,
                    headers=headers,
                    timeout=self.valves.REQUEST_TIMEOUT,
                ) as client:
                    resp = await client.post(
                        "/models/gemini-2.5-flash-image-preview:generateContent",
                        json=payload,
                    )
                    if resp.status_code != 200:
                        yield self._format_data(
                            is_stream=is_stream,
                            content=f"API error: {resp.status_code} - {resp.text}",
                        )
                        return
                    response_data = resp.json()

                    parts = (
                        response_data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [])
                    )
                    image_part = next((p for p in parts if "inlineData" in p), None)
                    if not image_part:
                        yield self._format_data(
                            is_stream=is_stream,
                            content="Error: No image generated in response.",
                        )
                        return

                    mime_type = image_part["inlineData"]["mimeType"]
                    base64_data = image_part["inlineData"]["data"]
                    text_part = next(
                        (p.get("text", "") for p in parts if "text" in p), ""
                    )

                    await _send_status("Image generated, uploading to your library…")
                    image_url = self._upload_image(
                        __request__=__request__,
                        user=user,
                        image_data=base64_data,
                        mime_type=mime_type,
                    )
                    final_content = (
                        f"{text_part}\n\n![Generated Image]({image_url})"
                        if text_part
                        else f"![Generated Image]({image_url})"
                    )
                    if is_stream:
                        yield self._format_data(
                            is_stream=True, model=model, content=final_content
                        )
                    else:
                        yield self._format_data(
                            is_stream=False, model=model, content=final_content
                        )

            except Exception as e:
                yield self._format_data(is_stream=is_stream, content=f"Error: {e}")

        return StreamingResponse(stream_response())

    # ------------------------------------------------------------------
    # Helper functions — unchanged except removal of logger lines
    # ------------------------------------------------------------------
    async def _find_image(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        max_search = 5
        search_count = 0
        for msg in reversed(messages):
            if search_count >= max_search:
                break
            search_count += 1
            content = msg.get("content", [])
            if isinstance(content, str):
                match = re.search(
                    r"!\[([^\]]*)\]\((data:image[^;]+;base64,[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)",
                    content,
                )
                if match:
                    url = match.group(2)
                    if url.startswith("data:"):
                        return url
                    elif "/files/" in url or "/api/v1/files/" in url:
                        b64 = await self._fetch_file_as_base64(url)
                        if b64:
                            return b64
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            return url
                        elif "/files/" in url or "/api/v1/files/" in url:
                            b64 = await self._fetch_file_as_base64(url)
                            if b64:
                                return b64
                    elif item.get("type") == "text":
                        text = item.get("text", "")
                        match = re.search(
                            r"!\[([^\]]*)\]\((data:image[^;]+;base64,[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)",
                            text,
                        )
                        if match:
                            url = match.group(2)
                            if url.startswith("data:"):
                                return url
                            elif "/files/" in url or "/api/v1/files/" in url:
                                b64 = await self._fetch_file_as_base64(url)
                                if b64:
                                    return b64
        return None

    async def _fetch_file_as_base64(self, file_url: str) -> Optional[str]:
        try:
            if "/api/v1/files/" in file_url:
                fid = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files/")[-1].split("/")[0].split("?")[0]

            from open_webui.models.files import Files
            import aiofiles

            file_obj = Files.get_file_by_id(fid)
            if file_obj and file_obj.path:
                async with aiofiles.open(file_obj.path, "rb") as fp:
                    raw = await fp.read()
                enc = base64.b64encode(raw).decode()
                mime = file_obj.meta.get("content_type", "image/png")
                return f"data:{mime};base64,{enc}"
        except Exception:
            pass
        return None

    def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        bio = io.BytesIO(base64.b64decode(image_data))
        bio.seek(0)
        up_obj = upload_file(
            request=__request__,
            background_tasks=BackgroundTasks(),
            file=UploadFile(
                file=bio,
                filename=f"gen-img-{uuid.uuid4().hex}.png",
                headers=Headers({"content-type": mime_type}),
            ),
            process=False,
            user=user,
            metadata={"mime_type": mime_type},
        )
        return __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)

    def _format_data(
        self,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
    ) -> str:
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if content:
            data["choices"] = [
                {
                    "finish_reason": "stop" if not is_stream else None,
                    "index": 0,
                    "delta" if is_stream else "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ]
        return f"data: {json.dumps(data)}\n\n" if is_stream else json.dumps(data) + "\n"
