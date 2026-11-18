"""Allume, éteint ou décrit la démo cloud : Inference Endpoint (vLLM, GPU, payant) et Space de l'API (CPU, gratuit).

    uv run python scripts/cloud.py status     état de l'endpoint et du Space, puis /health de l'API
    uv run python scripts/cloud.py start      relance l'endpoint et le Space, attend qu'ils soient prêts (~5-10 min)
    uv run python scripts/cloud.py stop       met l'endpoint en pause : plus aucune facturation GPU

L'endpoint en pause n'est pas facturé ; le Space (CPU gratuit) reste allumé et se met en veille seul après 48 h.
Nom de l'endpoint : variable HF_ENDPOINT_NAME (défaut ci-dessous). Jeton : connexion locale (`hf auth login`).
"""

import os
import sys
import time

import httpx
from huggingface_hub import HfApi, get_inference_endpoint, get_token

SPACE_ID = "0biyohan/triage-api"
SPACE_URL = "https://0biyohan-triage-api.hf.space"
ENDPOINT_NAME = os.environ.get("HF_ENDPOINT_NAME", "qwen3-medical-qcm-fr")


def endpoint_or_none():
    try:
        return get_inference_endpoint(ENDPOINT_NAME)
    except Exception as error:  # endpoint pas encore créé, ou nom différent
        print(f"Endpoint « {ENDPOINT_NAME} » introuvable ({type(error).__name__}) : définir HF_ENDPOINT_NAME ?")
        return None


def api_health() -> dict:
    try:
        response = httpx.get(f"{SPACE_URL}/health", headers={"Authorization": f"Bearer {get_token()}"}, timeout=15)
        return response.json() if response.status_code == 200 else {"http": response.status_code}
    except httpx.HTTPError as error:
        return {"erreur": type(error).__name__}


def status() -> None:
    endpoint = endpoint_or_none()
    if endpoint:
        print(f"Endpoint {endpoint.name} : {endpoint.status}  ({endpoint.url or 'pas d URL : en pause ou en veille'})")
    print(f"Space {SPACE_ID} : {HfApi().get_space_runtime(SPACE_ID).stage}")
    print(f"API {SPACE_URL}/health : {api_health()}")


def start() -> None:
    api = HfApi()
    if api.get_space_runtime(SPACE_ID).stage != "RUNNING":
        print("Space : redémarrage…")
        api.restart_space(SPACE_ID)
    endpoint = endpoint_or_none()
    if endpoint:
        if endpoint.status == "paused":
            print("Endpoint : reprise (facturation GPU à partir de maintenant)…")
            endpoint.resume()
        elif endpoint.status == "scaledToZero":
            print("Endpoint : réveil depuis la mise en veille…")
            endpoint.resume()
        endpoint.wait(timeout=900)
        print(f"Endpoint prêt : {endpoint.url}")
    for _ in range(60):  # le Space peut mettre une minute à démarrer
        if api.get_space_runtime(SPACE_ID).stage == "RUNNING":
            break
        time.sleep(10)
    status()


def stop() -> None:
    endpoint = endpoint_or_none()
    if endpoint:
        endpoint.pause()
        print(f"Endpoint {endpoint.name} en pause : plus de facturation GPU. Vérifier la page Billing après la 1re fois.")


if __name__ == "__main__":
    commands = {"status": status, "start": start, "stop": stop}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
