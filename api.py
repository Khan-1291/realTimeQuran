from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr import FasterWhisperASR
from matcher import MatchResult, RecitationMatcher
from quran_loader import QuranEntry, build_surah_summaries, ensure_quran_dataset, get_surah_entries, load_quran
from stream_audio import decode_audio_bytes


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "quran.txt"
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"


class SessionCreateRequest(BaseModel):
    surah: int = Field(..., ge=1)
    start_ayah: int = Field(default=1, ge=1)


class SurahResponse(BaseModel):
    surah: int
    name: str
    ayah_count: int


class AyahResponse(BaseModel):
    surah: int
    ayah: int
    text: str


class SessionResponse(BaseModel):
    session_id: str
    surah: int
    current_ayah: int
    expected_text: str
    recognized_text: str
    similarity: float
    status: str
    missing_words: list[str]
    incorrect_pairs: list[str]
    is_complete: bool
    session_complete: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class RecitationSession:
    session_id: str
    surah: int
    target_ayah: int
    matcher: RecitationMatcher
    created_at: datetime
    updated_at: datetime
    audio_buffer: bytearray

    def to_response(self, result: MatchResult | None = None) -> SessionResponse:
        current = result or self.matcher.current_state()
        return SessionResponse(
            session_id=self.session_id,
            surah=self.surah,
            current_ayah=current.ayah,
            expected_text=current.expected_text,
            recognized_text=self.matcher.recognized_text,
            similarity=current.similarity,
            status=current.status,
            missing_words=current.missing_words,
            incorrect_pairs=current.incorrect_pairs,
            is_complete=current.is_complete,
            session_complete=current.session_complete,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class RecitationService:
    def __init__(self, model_size: str = "small") -> None:
        ensure_quran_dataset(DATASET_PATH, source_root=BASE_DIR / "dataset")
        self.entries = load_quran(DATASET_PATH)
        self.summaries = build_surah_summaries(self.entries)
        self.asr = FasterWhisperASR(model_size=model_size, model_dir=MODEL_DIR)
        self.sessions: dict[str, RecitationSession] = {}
        self._lock = asyncio.Lock()

    async def preload(self) -> None:
        await asyncio.to_thread(self.asr.load)

    def list_surahs(self) -> list[SurahResponse]:
        return [SurahResponse(**summary.__dict__) for summary in self.summaries]

    def list_ayahs(self, surah: int) -> list[AyahResponse]:
        return [
            AyahResponse(surah=entry.surah, ayah=entry.ayah, text=entry.text)
            for entry in get_surah_entries(self.entries, surah)
        ]

    async def create_session(self, surah: int, start_ayah: int) -> SessionResponse:
        async with self._lock:
            surah_entries = get_surah_entries(self.entries, surah)
            valid_ayahs = {entry.ayah for entry in surah_entries}
            if start_ayah not in valid_ayahs:
                raise HTTPException(status_code=400, detail="Invalid starting ayah for selected surah")

            now = datetime.now(UTC)
            session = RecitationSession(
                session_id=uuid.uuid4().hex,
                surah=surah,
                target_ayah=start_ayah,
                matcher=RecitationMatcher(
                    [entry for entry in surah_entries if entry.ayah == start_ayah],
                    start_surah=surah,
                    start_ayah=start_ayah,
                ),
                created_at=now,
                updated_at=now,
                audio_buffer=bytearray(),
            )
            self.sessions[session.session_id] = session
            return session.to_response()

    def get_session(self, session_id: str) -> RecitationSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def process_audio_chunk(self, session_id: str, audio_bytes: bytes) -> SessionResponse:
        session = self.get_session(session_id)
        session.audio_buffer.extend(audio_bytes)
        try:
            audio = await asyncio.to_thread(decode_audio_bytes, bytes(session.audio_buffer))
        except Exception:
            session.updated_at = datetime.now(UTC)
            return session.to_response()
        if audio.size == 0:
            session.updated_at = datetime.now(UTC)
            return session.to_response()
        expected_prompt = session.matcher.current_entry.text
        transcript = await asyncio.to_thread(
            self.asr.transcribe_chunk,
            audio,
            expected_prompt,
        )
        result = session.matcher.update(
            transcript.text,
            transcript.word_timestamps,
            replace_recognized_text=True,
        )
        session.updated_at = datetime.now(UTC)
        return session.to_response(result)


def create_app() -> FastAPI:
    app = FastAPI(title="Quran Recitation Checker API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("QURAN_API_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = RecitationService(model_size=os.getenv("QURAN_ASR_MODEL", "small"))
    app.state.recitation_service = service
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.on_event("startup")
    async def startup() -> None:
        await service.preload()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/api")
    async def api_root() -> dict[str, object]:
        return {
            "name": "Quran Recitation Checker API",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "surahs": "/surahs",
        }

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/surahs", response_model=list[SurahResponse])
    async def list_surahs() -> list[SurahResponse]:
        return service.list_surahs()

    @app.get("/surahs/{surah}/ayahs", response_model=list[AyahResponse])
    async def list_ayahs(surah: int) -> list[AyahResponse]:
        try:
            return service.list_ayahs(surah)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions", response_model=SessionResponse)
    async def create_session(payload: SessionCreateRequest) -> SessionResponse:
        return await service.create_session(payload.surah, payload.start_ayah)

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str) -> SessionResponse:
        session = service.get_session(session_id)
        return session.to_response()

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, str]:
        await service.delete_session(session_id)
        return {"status": "deleted"}

    @app.post("/sessions/{session_id}/audio", response_model=SessionResponse)
    async def process_audio(session_id: str, request: Request) -> SessionResponse:
        audio_bytes = await request.body()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Request body does not contain audio bytes")
        try:
            return await service.process_audio_chunk(session_id, audio_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/ws/sessions/{session_id}")
    async def recitation_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            session = service.get_session(session_id)
        except HTTPException:
            await websocket.send_json({"type": "error", "detail": "Session not found"})
            await websocket.close(code=4404)
            return

        await websocket.send_json({"type": "session_ready", **session.to_response().model_dump(mode="json")})

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    try:
                        response = await service.process_audio_chunk(session_id, message["bytes"])
                        await websocket.send_json({"type": "update", **response.model_dump(mode="json")})
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                elif "text" in message and message["text"] is not None:
                    if message["text"].strip().lower() == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        await websocket.send_json(
                            {"type": "error", "detail": "Send binary audio chunks or 'ping'"}
                        )
        except WebSocketDisconnect:
            return

    return app


app = create_app()
