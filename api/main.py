"""API de démonstration : réponse à un QCM médical par le modèle Qwen3-1.7B adapté (SFT + DPO), servi par vLLM.

Le prompt est construit avec `projet_14.mediqal`, comme à l'entraînement : le modèle reçoit exactement le format appris.
Chaque requête est tracée dans un journal d'audit JSONL, sans le texte du cas (seulement son empreinte SHA-256).

Variables d'environnement :
    API_KEY     clé attendue dans l'en-tête X-API-Key (obligatoire)
    VLLM_URL    adresse du serveur vLLM (défaut : http://vllm:8000)
    MODEL_NAME  nom du modèle servi par vLLM (défaut : qwen3-medical)
    AUDIT_LOG   chemin du journal d'audit (défaut : logs/audit.jsonl)
"""

import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from projet_14.mediqal import LETTERS, SYSTEM_PROMPT, answer_letters, user_content

WARNING = "Démonstration technique : ne remplace pas un avis médical."

app = FastAPI(title="Agent de triage médical — démonstration", version="0.1.0")


def settings() -> dict:
    # Lu à chaque requête : la configuration vient de l'environnement (docker compose, tests)
    return {
        "api_key": os.environ.get("API_KEY", ""),
        "vllm_url": os.environ.get("VLLM_URL", "http://vllm:8000"),
        "model": os.environ.get("MODEL_NAME", "qwen3-medical"),
        "audit_log": Path(os.environ.get("AUDIT_LOG", "logs/audit.jsonl")),
    }


def check_api_key(key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> None:
    expected = settings()["api_key"]
    if not expected or not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")


class QCMRequest(BaseModel):
    cas_clinique: str | None = Field(default=None, max_length=10_000)
    question: str = Field(min_length=1, max_length=2_000)
    propositions: dict[Literal["A", "B", "C", "D", "E"], str] = Field(min_length=2)
    reponses_multiples: bool = False


class QCMResponse(BaseModel):
    request_id: str
    lettres: list[str]
    reponse: str
    modele: str
    latence_ms: int
    avertissement: str = WARNING


def build_messages(request: QCMRequest) -> list[dict]:
    # Même structure que les données d'entraînement (colonnes MediQAl) : même prompt, consigne comprise
    row = {"clinical_case": request.cas_clinique, "question": request.question}
    row |= {f"answer_{letter.lower()}": request.propositions.get(letter, "") for letter in LETTERS}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(row, multiple_answers=request.reponses_multiples)},
    ]


async def call_vllm(messages: list[dict], max_tokens: int) -> str:
    config = settings()
    payload = {"model": config["model"], "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{config['vllm_url']}/v1/chat/completions", json=payload)
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def write_audit(record: dict) -> None:
    path = settings()["audit_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/health")
async def health() -> dict:
    # L'API répond ; vLLM est-il prêt ?
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            vllm_ready = (await client.get(f"{settings()['vllm_url']}/health")).status_code == 200
    except httpx.HTTPError:
        vllm_ready = False
    return {"api": "ok", "vllm": "ok" if vllm_ready else "indisponible"}


@app.post("/qcm", response_model=QCMResponse, dependencies=[Depends(check_api_key)])
async def qcm(request: QCMRequest) -> QCMResponse:
    request_id = str(uuid.uuid4())
    model = settings()["model"]
    start = time.perf_counter()
    audit = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        # Empreinte de l'entrée : retrouver une requête sans conserver de donnée patient
        "input_sha256": hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest(),
        "reponses_multiples": request.reponses_multiples,
    }
    try:
        answer = await call_vllm(build_messages(request), max_tokens=192 if request.reponses_multiples else 64)
    except httpx.HTTPError as error:
        write_audit(audit | {"status": "error", "error": type(error).__name__,
                             "latency_ms": round((time.perf_counter() - start) * 1000)})
        raise HTTPException(status_code=503, detail="Serveur d'inférence indisponible") from error

    latency_ms = round((time.perf_counter() - start) * 1000)
    letters = sorted(answer_letters(answer))
    write_audit(audit | {"status": "ok", "letters": letters, "latency_ms": latency_ms})
    return QCMResponse(request_id=request_id, lettres=letters, reponse=answer, modele=model, latence_ms=latency_ms)
