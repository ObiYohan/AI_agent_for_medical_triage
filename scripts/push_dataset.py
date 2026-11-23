"""Publie le dataset anonymisé (SFT v4 + DPO v5) sur le Hub Hugging Face et pose une étiquette de version.

    uv run python scripts/push_dataset.py --tag v1.0

N'envoie que les fichiers anonymisés et leurs rapports (comptes seulement), jamais les échantillons de contrôle RGPD.
Chaque envoi remplace le contenu précédent ; l'étiquette fige la version (`load_dataset(..., revision="v1.0")`).
Le jeton Hugging Face (écriture) vient de la variable HF_TOKEN, ou de la connexion locale (`hf auth login`).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "0biyohan/triage-medical-bilingual"
SOURCES = {"sft": ROOT / "data/v4/anonymized", "dpo": ROOT / "data/v5/anonymized"}  # dossier du Hub -> données
CARD = ROOT / "deploy/dataset/README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="étiquette de version, par exemple v1.0")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for folder, source in SOURCES.items():
            files = sorted(source.glob(f"{folder}_*.anonymized.jsonl"))
            if not files:
                sys.exit(f"Aucun fichier {folder}_*.anonymized.jsonl dans {source} : lancer les notebooks de préparation.")
            (Path(tmp) / folder).mkdir()
            (Path(tmp) / "anonymization").mkdir(exist_ok=True)
            for file in files:
                shutil.copy(file, Path(tmp) / folder / file.name.replace(".anonymized", ""))
            for report in source.glob(f"{folder}_*.report.json"):
                shutil.copy(report, Path(tmp) / "anonymization" / report.name)
        shutil.copy(CARD, Path(tmp) / "README.md")
        commit = api.upload_folder(
            folder_path=tmp, repo_id=REPO_ID, repo_type="dataset",
            delete_patterns=["*.jsonl", "*.json"],  # supprime les fichiers distants absents de l'envoi
            commit_message=f"Dataset {args.tag}",
        )
    api.create_tag(REPO_ID, tag=args.tag, repo_type="dataset", revision=commit.oid)
    print(f"{REPO_ID} @ {args.tag} : {commit.commit_url}")


if __name__ == "__main__":
    main()
