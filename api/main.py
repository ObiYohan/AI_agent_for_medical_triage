"""API de démonstration : triage d'un patient aux urgences par le modèle Qwen3-1.7B adapté (SFT + DPO), servi par vLLM.

Le patient est décrit en langage naturel (âge, motif, symptômes, antécédents, constantes) ; l'API renvoie un niveau
de priorité, les signes de gravité, une justification, la conduite à tenir et des questions complémentaires
(voir `triage.py`). Le prompt reprend le format de l'entraînement (`projet_14.mediqal`).
Chaque requête est tracée dans un journal d'audit JSONL, sans le texte du patient (seulement son empreinte SHA-256).

Variables d'environnement :
    API_KEY       clé attendue dans l'en-tête X-API-Key (obligatoire)
    VLLM_URL      adresse du serveur vLLM (défaut : http://vllm:8000) ; local ou distant (Inference Endpoint)
    VLLM_API_KEY  jeton envoyé à vLLM (Authorization: Bearer), pour un serveur distant protégé (défaut : aucun)
    VLLM_TIMEOUT  délai maximal d'une génération, en secondes (défaut : 60 ; 300 pour couvrir un réveil)
    MODEL_NAME    nom du modèle servi par vLLM (défaut : qwen3-medical)
    TRIAGE_CALIBRATION  0 (défaut) : niveau brut du modèle ; 1 : niveau corrigé de son a priori (voir triage.py).
                  Expérimental : sur 60 cas externes, elle sous-trie plus de la moitié des urgences vitales
    AUDIT_LOG     chemin du journal d'audit (défaut : logs/audit.jsonl), aussi écrit sur la sortie standard
    UI_PASSWORD   active l'interface web (Gradio, `web_ui.py`) sous /ui, avec ce mot de passe (démo cloud ; Gradio
                  requis : deploy/space/requirements-ui.txt). UI_USER : identifiant (défaut : jury). TRIAGE_CASES :
                  cas de démonstration proposés dans l'interface (défaut : triage_cases.json à côté de l'API)
"""

import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

import triage as tri
from projet_14.mediqal import SYSTEM_PROMPT, user_content

WARNING = "Démonstration technique : ne remplace pas un avis médical."

app = FastAPI(title="Agent de triage médical — démonstration", version="0.2.0")


def settings() -> dict:
    # Lu à chaque requête : la configuration vient de l'environnement (docker compose, Space, tests)
    return {
        "api_key": os.environ.get("API_KEY", ""),
        "vllm_url": os.environ.get("VLLM_URL", "http://vllm:8000"),
        "vllm_api_key": os.environ.get("VLLM_API_KEY", ""),
        "vllm_timeout": float(os.environ.get("VLLM_TIMEOUT", "60")),
        "model": os.environ.get("MODEL_NAME", "qwen3-medical"),
        "calibration": os.environ.get("TRIAGE_CALIBRATION", "0") == "1",
        "audit_log": Path(os.environ.get("AUDIT_LOG", "logs/audit.jsonl")),
    }


def vllm_headers(config: dict) -> dict:
    # Jeton pour un vLLM distant (Inference Endpoint privé) ; aucun en local
    return {"Authorization": f"Bearer {config['vllm_api_key']}"} if config["vllm_api_key"] else {}


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def valid_key(key: str | None) -> bool:
    expected = settings()["api_key"]
    return bool(expected and key and secrets.compare_digest(key, expected))


def check_api_key(key: str | None = Security(API_KEY_HEADER)) -> None:
    if not valid_key(key):
        # Accès refusé : tracé aussi (sans la clé reçue)
        write_audit({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "unauthorized"})
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")


async def call_vllm(messages: list[dict]) -> list[dict]:
    # Un seul token (la lettre du niveau) et ses 20 alternatives les plus probables, avec leurs log-probabilités
    config = settings()
    payload = {"model": config["model"], "messages": messages, "temperature": 0, "max_tokens": 1,
               "logprobs": True, "top_logprobs": 20}
    # X-Scale-Up-Timeout : un Inference Endpoint en veille fait patienter la requête pendant son réveil, au lieu de
    # répondre 503 (en-tête ignoré par un vLLM local)
    headers = vllm_headers(config) | {"X-Scale-Up-Timeout": str(int(config["vllm_timeout"]))}
    async with httpx.AsyncClient(timeout=config["vllm_timeout"]) as client:
        response = await client.post(f"{config['vllm_url']}/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
    return response.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


def level_messages(request: tri.TriageRequest | None) -> list[dict]:
    # Prompt du triage ; sans patient, celui du cas vide de la calibration
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content(tri.prompt_row(request), multiple_answers=False)}]


PRIORS: dict[tuple[str, str], dict[str, float]] = {}  # a priori du modèle, par serveur et modèle (calculé une fois)


async def model_prior() -> dict[str, float] | None:
    config = settings()
    if not config["calibration"]:
        return None
    key = (config["vllm_url"], config["model"])
    if key not in PRIORS:
        probabilities = tri.level_probabilities(await call_vllm(level_messages(None)))
        if probabilities is None:
            return None
        PRIORS[key] = probabilities
    return PRIORS[key]


def write_audit(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    path = settings()["audit_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    # Aussi sur la sortie standard : visible dans les logs de la plateforme (Space, conteneur), même si le disque est
    # effacé au redémarrage
    print(f"AUDIT {line}", flush=True)


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


@app.get("/health")
async def health(key: str | None = Security(API_KEY_HEADER)) -> dict:
    # L'API répond ; vLLM est-il prêt ? vLLM n'est interrogé qu'avec une clé valide : /health est ouvert à tous, et
    # une requête vers un Inference Endpoint en veille le réveille (facturé). Sans clé, seul l'état de l'API est donné.
    # Sans X-Scale-Up-Timeout ni long délai : un endpoint en veille est signalé « indisponible » sans attendre son réveil
    if not valid_key(key):
        return {"api": "ok"}
    config = settings()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{config['vllm_url']}/health", headers=vllm_headers(config))
            vllm_ready = response.status_code == 200
    except httpx.HTTPError:
        vllm_ready = False
    return {"api": "ok", "vllm": "ok" if vllm_ready else "indisponible"}


@app.post("/triage", response_model=tri.TriageResponse, dependencies=[Depends(check_api_key)])
async def triage(request: tri.TriageRequest) -> tri.TriageResponse:
    """Triage d'un patient décrit en langage naturel (âge, motif, symptômes, antécédents, constantes).

    Le modèle évalue le niveau de priorité sur un choix fermé ; des garde-fous sur les constantes ne peuvent que
    relever ce niveau. La réponse explique la décision et propose des questions complémentaires.
    """
    request_id = str(uuid.uuid4())
    model = settings()["model"]
    start = time.perf_counter()
    flags = tri.red_flags(request.constantes)
    audit = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "triage",
        "model": model,
        # Empreinte de l'entrée : retrouver une requête sans conserver de donnée patient
        "input_sha256": hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest(),
        "red_flags": len(flags),
    }
    model_level = raw_level = None
    try:
        probabilities = tri.level_probabilities(await call_vllm(level_messages(request)))
        if probabilities is not None:
            raw_level, model_level = tri.model_levels(probabilities, await model_prior())
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 400:
            # vLLM refuse la requête : le texte dépasse la longueur maximale du modèle (--max-model-len)
            write_audit(audit | {"status": "rejected", "error": "input_too_long", "latency_ms": elapsed_ms(start)})
            raise HTTPException(status_code=422, detail="Description trop longue pour le modèle") from error
        unavailable(audit, start, error, flags)
    except httpx.HTTPError as error:
        unavailable(audit, start, error, flags)

    benign, benign_reason = tri.benign_assessment(request)
    level, source = tri.decide(model_level, flags, benign)
    latency_ms = elapsed_ms(start)
    write_audit(audit | {"status": "ok", "niveau": level, "source": source, "niveau_modele": model_level,
                         "niveau_modele_brut": raw_level, "latency_ms": latency_ms})
    return tri.TriageResponse(
        request_id=request_id, niveau=level, source=source, niveau_modele=model_level, niveau_modele_brut=raw_level,
        signes_de_gravite=flags, justification=tri.justification(level, source, model_level, flags, raw_level, benign,
                                                                benign_reason),
        conduite=tri.CONDUITE[level],
        questions_complementaires=tri.follow_up_questions(request), modele=model, latence_ms=latency_ms,
        avertissement=WARNING,
    )


def unavailable(audit: dict, start: float, error: Exception, flags: list[str]) -> None:
    # Modèle indisponible : avec un signe de gravité, les garde-fous suffisent à décider (on continue) ; sinon, 503
    if flags:
        return
    write_audit(audit | {"status": "error", "error": type(error).__name__, "latency_ms": elapsed_ms(start)})
    raise HTTPException(status_code=503, detail="Serveur d'inférence indisponible") from error


# Interface web de démonstration (Space) : seulement si un mot de passe est défini
if os.environ.get("UI_PASSWORD"):
    import web_ui

    async def ui_health() -> dict:
        return await health(settings()["api_key"])  # état de vLLM compris : l'utilisateur est authentifié

    app = web_ui.mount(
        app, triage, ui_health,
        cases=web_ui.load_cases(Path(os.environ.get("TRIAGE_CASES", Path(__file__).with_name("triage_cases.json")))),
        user=os.environ.get("UI_USER", "jury"), password=os.environ["UI_PASSWORD"],
    )
