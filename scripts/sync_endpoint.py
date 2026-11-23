"""Aligne l'Inference Endpoint (démo cloud) sur le modèle servi en local : même dépôt, même révision.

    uv run python scripts/sync_endpoint.py             met à jour l'endpoint si son modèle diffère
    uv run python scripts/sync_endpoint.py --dry-run   affiche seulement l'écart

La référence est `docker-compose.yml` (`--model=${MODEL:-…}`, `--revision=${MODEL_REVISION:-…}`) : changer la révision
du modèle dans ce fichier, puis pousser sur `master`, met à jour la pile locale et l'endpoint (job `deploy-cloud`).

Facturation : un endpoint en pause le reste (il est remis en pause après la mise à jour) ; un endpoint allumé ou en
veille redémarre sur le nouveau modèle. Rien n'est fait si l'endpoint est déjà à jour.
Jeton : variable HF_TOKEN (droit d'écriture sur les Inference Endpoints), ou connexion locale (`hf auth login`).
"""

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import get_inference_endpoint

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_NAME = os.environ.get("HF_ENDPOINT_NAME", "qwen3-medical-qcm-fr")


def compose_model() -> tuple[str, str]:
    # Valeurs par défaut de docker-compose.yml : le modèle et la révision servis par la pile locale
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    model = re.search(r"--model=\$\{MODEL:-([^}]+)\}", text)
    revision = re.search(r"--revision=\$\{MODEL_REVISION:-([0-9a-f]{40})\}", text)
    if not (model and revision):
        sys.exit("Modèle ou révision introuvable dans docker-compose.yml")
    return model.group(1), revision.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="afficher l'écart sans rien modifier")
    args = parser.parse_args()

    repository, revision = compose_model()
    endpoint = get_inference_endpoint(ENDPOINT_NAME, token=os.environ.get("HF_TOKEN") or None)
    print(f"docker-compose.yml : {repository} @ {revision[:7]}")
    print(f"endpoint {ENDPOINT_NAME} : {endpoint.repository} @ {(endpoint.revision or '?')[:7]} ({endpoint.status})")
    if (endpoint.repository, endpoint.revision) == (repository, revision):
        print("Endpoint à jour : rien à faire.")
        return
    if args.dry_run:
        print("Écart (--dry-run : aucune modification).")
        return

    was_paused = endpoint.status == "paused"
    endpoint = endpoint.update(repository=repository, revision=revision)
    print(f"Endpoint mis à jour : {endpoint.repository} @ {endpoint.revision[:7]} ({endpoint.status})")
    if was_paused and endpoint.status != "paused":
        endpoint = endpoint.pause()  # il était en pause : pas de facturation due à la mise à jour
        print(f"Remis en pause ({endpoint.status}).")


if __name__ == "__main__":
    main()
