"""Interface web de test de l'endpoint de triage (/triage), en local comme sur le cloud.

    uv run --with gradio python scripts/ui.py                 puis ouvrir http://127.0.0.1:7860

Même interface que celle du Space (`api/web_ui.py`), avec en plus le choix de la cible (pile locale, Space Hugging
Face ou autre URL), appelée par HTTP. Les secrets restent côté serveur Gradio, jamais dans la page : clé de l'API lue
dans API_KEY ou dans .env, jeton Hugging Face (Space privé) dans HF_TOKEN ou dans la connexion locale
(`hf auth login`). L'interface n'écoute que sur la machine (127.0.0.1) : pas de lien public, puisqu'elle appelle l'API
avec la clé.
"""

import os
import sys
from pathlib import Path

import httpx
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
import web_ui  # noqa: E402  (interface partagée avec le Space)

TARGETS = {
    "Local (docker compose)": "http://localhost:8080",
    "Cloud (Space Hugging Face)": "https://0biyohan-triage-api.hf.space",
}
# Réponse : jusqu'à 300 s (réveil de l'Inference Endpoint) ; connexion : 10 s (URL fausse ou pile arrêtée)
TIMEOUT = httpx.Timeout(300, connect=10)


def api_key() -> str:
    # Variable d'environnement, sinon la seule ligne API_KEY du fichier .env (le reste du fichier n'est pas lu)
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def headers(url: str) -> dict[str, str]:
    result = {"X-API-Key": api_key()}
    if ".hf.space" in url:  # Space privé : le jeton Hugging Face ouvre l'accès au Space, la clé reste exigée par l'API
        token = os.environ.get("HF_TOKEN") or get_token()
        if token:
            result["Authorization"] = f"Bearer {token}"
    return result


def target_url(target: str, custom: str) -> str:
    return (custom.strip() or TARGETS[target]).rstrip("/")


def health(target: str, custom: str) -> str:
    url = target_url(target, custom)
    try:
        response = httpx.get(f"{url}/health", headers=headers(url), timeout=httpx.Timeout(30, connect=10))
    except httpx.HTTPError as error:
        return f"❌ `{url}` injoignable : {type(error).__name__}"
    if response.status_code != 200:
        return f"❌ `{url}` : HTTP {response.status_code}"
    body = response.json()
    state = "✅" if body and all(value == "ok" for value in body.values()) else "⚠️"  # {"api": "ok", "vllm": "ok"}
    return f"{state} `{url}` : " + ", ".join(f"{key} = {value}" for key, value in body.items())


def triage(target, custom, *fields):
    url = target_url(target, custom)
    request = web_ui.build_request(*fields)
    try:
        response = httpx.post(f"{url}/triage", json=request, headers=headers(url), timeout=TIMEOUT)
    except httpx.HTTPError as error:
        return f"❌ `{url}` injoignable : {type(error).__name__}", request
    try:
        body = response.json()
    except ValueError:
        body = {"texte": response.text[:500]}
    if response.status_code == 200:
        return web_ui.render(body), body
    if response.status_code == 422 and isinstance(body.get("detail"), list):  # champs refusés par la validation
        return web_ui.refusal([(e.get("loc", [])[1:], e.get("msg")) for e in body["detail"]]), body
    return f"❌ HTTP {response.status_code} : {body.get('detail', body)}", body


demo = web_ui.build(triage, health, web_ui.load_cases(ROOT / "scripts" / "triage_cases.json"), targets=TARGETS,
                    subtitle="Test de l'endpoint `/triage`, en local ou sur le cloud.")


if __name__ == "__main__":
    if not api_key():
        print("Attention : pas de clé d'API (variable API_KEY ou ligne API_KEY= dans .env) ; l'API répondra 401.")
    demo.launch(server_name="127.0.0.1")
