"""Déploie l'API sur le Space Hugging Face `0biyohan/triage-api` (Docker, CPU gratuit).

Assemble le contenu du Space (Dockerfile et README de `deploy/space/`, code de l'API, module partagé du prompt) et
l'envoie en un seul commit, qui remplace l'ancien contenu. Le Space reconstruit alors son image.

    uv run --no-project --with huggingface_hub python scripts/deploy_space.py [--revision <commit git>] [--wait]

Avec --wait : attend que le Space fasse tourner ce nouveau commit, puis vérifie GET /health (test de fumée).
Le jeton Hugging Face (écriture sur le Space) vient de la variable HF_TOKEN, ou de la connexion locale (`hf auth login`).
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "0biyohan/triage-api"
SPACE_URL = "https://0biyohan-triage-api.hf.space"
FILES = {  # chemin dans le Space -> fichier du dépôt
    "Dockerfile": "deploy/space/Dockerfile",
    "README.md": "deploy/space/README.md",
    "requirements.txt": "api/requirements.txt",
    "main.py": "api/main.py",
    "triage.py": "api/triage.py",
    "projet_14/__init__.py": "src/projet_14/__init__.py",
    "projet_14/mediqal.py": "src/projet_14/mediqal.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default="local", help="commit git déployé (message du commit du Space)")
    parser.add_argument("--wait", action="store_true", help="attendre la nouvelle version, puis tester /health")
    args = parser.parse_args()

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        for target, source in FILES.items():
            (Path(tmp) / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / source, Path(tmp) / target)
        commit = api.upload_folder(
            folder_path=tmp, repo_id=SPACE_ID, repo_type="space",
            delete_patterns=["*"],  # supprime les fichiers distants absents de l'envoi (ancien contenu)
            commit_message=f"Deploy API ({args.revision})",
        )
    print(f"Space {SPACE_ID} mis à jour : {commit.commit_url}")
    if args.wait:
        wait_and_check(api, commit.oid)


def wait_and_check(api: HfApi, sha: str, timeout: int = 900) -> None:
    # Le Space reconstruit son image : attendre qu'il fasse tourner CE commit (pas l'ancienne version encore active)
    deadline = time.time() + timeout
    while True:
        runtime = api.get_space_runtime(SPACE_ID)
        if runtime.raw.get("sha") == sha and runtime.stage == "RUNNING":
            break
        if runtime.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            sys.exit(f"Échec du Space : {runtime.stage} (voir ses logs)")
        if time.time() > deadline:
            sys.exit(f"Le Space ne fait pas tourner {sha[:7]} après {timeout} s (état : {runtime.stage})")
        time.sleep(10)
    # Test de fumée : l'API répond. Sans clé d'API, /health n'interroge pas l'endpoint : il n'est pas réveillé
    response = httpx.get(f"{SPACE_URL}/health", headers={"Authorization": f"Bearer {get_token()}"}, timeout=30)
    print(f"GET /health : {response.status_code} {response.text}")
    if response.status_code != 200 or response.json().get("api") != "ok":
        sys.exit("Test de fumée échoué")


if __name__ == "__main__":
    main()
