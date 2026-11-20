"""Crée l'Inference Endpoint de la démo cloud : moteur vLLM, GPU T4, modèle fusionné à une révision fixée.

    uv run python scripts/create_endpoint.py

ATTENTION : la création démarre l'endpoint, donc la facturation (0,50 $/h sur T4). Le script attend qu'il soit
prêt, vérifie /v1/models, puis le met en pause (plus de facturation). Ensuite : `scripts/cloud.py start | stop`.

Prérequis : crédits sur le compte (huggingface.co/settings/billing) et un jeton autorisé à gérer les Inference
Endpoints (connexion `hf auth login`).
"""

import os

import httpx
from huggingface_hub import create_inference_endpoint, get_token

NAME = os.environ.get("HF_ENDPOINT_NAME", "qwen3-medical-qcm-fr")
REPOSITORY = "0biyohan/qwen3-1.7b-medical-qcm-fr"
REVISION = "4af91a7aa052c092cfb64568d441b84f8f4a88e8"  # même révision que docker-compose.yml (SFT v4 + DPO v5)
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"  # même version que la pile locale


def main() -> None:
    endpoint = create_inference_endpoint(
        NAME,
        repository=REPOSITORY,
        revision=REVISION,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor="aws",
        region="eu-west-1",  # Irlande
        instance_type="nvidia-t4",  # 16 Go, 0,50 $/h ; pas de bf16 : vLLM sert le modèle en float16
        instance_size="x1",
        type="authenticated",  # appel avec un jeton Hugging Face (VLLM_API_KEY de l'API) ; ex-« protected »
        min_replica=0,  # 0 réplique au repos : condition de la mise en veille automatique
        max_replica=1,
        scale_to_zero_timeout=15,  # mise en veille après 15 min sans requête (filet de sécurité)
        custom_image={"vLLM": {"url": VLLM_IMAGE, "healthRoute": "/health", "port": 8000, "tensorParallelSize": 1}},
        container_args=["--max-model-len", "2048", "--served-model-name", "qwen3-medical"],
    )
    print(f"Endpoint {NAME} créé, démarrage (facturation en cours)…")
    endpoint.wait(timeout=1200)
    url = endpoint.url  # l'URL ne change pas après une pause
    print(f"Prêt : {url}")

    models = httpx.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {get_token()}"}, timeout=60)
    print(f"GET /v1/models : {models.status_code} {models.text[:200]}")

    endpoint.pause()
    print("Endpoint en pause : plus de facturation. URL à mettre dans le secret VLLM_URL du Space :")
    print(url)


if __name__ == "__main__":
    main()
