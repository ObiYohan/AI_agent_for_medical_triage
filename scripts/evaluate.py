"""Évaluation d'un modèle servi par vLLM : QCM du jeu de test et contrôles de sécurité.

Le prompt est construit par la même fonction que l'API (`api/main.py`) : le modèle est évalué exactement comme il est
servi. Les requêtes partent une par une, pour limiter l'effet du regroupement des requêtes sur les réponses.

    uv run --with-requirements api/requirements.txt python scripts/evaluate.py --vllm-url http://localhost:8001 --label sft

Sorties : `reports/evaluation/<label>_answers.csv` (une ligne par question) et `<label>_summary.json`.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from main import QCMRequest, build_messages, valid_format  # noqa: E402  (même prompt que l'API)

from projet_14.mediqal import answer_letters, parse_user_content  # noqa: E402

DATA = ROOT / "data" / "anonymized"
OUT = ROOT / "reports" / "evaluation"
FIELDS = ("cas_clinique", "question", "propositions", "reponses_multiples")


def test_questions() -> list[dict]:
    # QCM du jeu de test MediQAl (mcqu et mcqm), avec leur corrigé
    questions = []
    for stem in ("sft_mediqal_mcqu", "sft_mediqal_mcqm"):
        with (DATA / f"{stem}_test.anonymized.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                questions.append(parse_user_content(record["messages"][1]["content"]) | {
                    "id": record["id"], "jeu": stem.removeprefix("sft_mediqal_"),
                    "attendu": sorted(answer_letters(record["messages"][2]["content"])), "dangereuses": []})
    return questions


def safety_questions() -> list[dict]:
    cases = json.loads((ROOT / "scripts" / "safety_cases.json").read_text(encoding="utf-8"))
    return [case | {"jeu": f"securite_{case['categorie']}", "reponses_multiples": False} for case in cases]


async def answer(client: httpx.AsyncClient, question: dict) -> dict:
    request = QCMRequest(**{key: question.get(key) for key in FIELDS if question.get(key) is not None})
    payload = {"model": "qwen3-medical", "messages": build_messages(request), "temperature": 0,
               "max_tokens": 192 if request.reponses_multiples else 64}
    response = await client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    letters = sorted(answer_letters(text))
    return {
        "id": question["id"], "jeu": question["jeu"], "attendu": "".join(question["attendu"]),
        "lettres": "".join(letters), "juste": letters == question["attendu"],
        "dangereuse": bool(set(letters) & set(question["dangereuses"])),
        "format_valide": valid_format(letters, request), "reponse": text,
    }


async def run(url: str, questions: list[dict]) -> pd.DataFrame:
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:
        rows = [await answer(client, q) for q in questions]  # une requête à la fois
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--label", required=True, help="nom du modèle évalué (sft, dpo…)")
    args = parser.parse_args()

    answers = asyncio.run(run(args.vllm_url, test_questions() + safety_questions()))
    summary = answers.groupby("jeu").agg(
        questions=("juste", "size"), exactitude=("juste", "mean"),
        reponses_dangereuses=("dangereuse", "sum"), format_valide=("format_valide", "mean"))
    print(summary.round(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    answers.to_csv(OUT / f"{args.label}_answers.csv", index=False)
    (OUT / f"{args.label}_summary.json").write_text(
        summary.reset_index().to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
