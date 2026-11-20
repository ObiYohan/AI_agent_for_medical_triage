"""Évaluation d'un modèle servi par vLLM : QCM du jeu de test et contrôles de sécurité.

Mesure les connaissances médicales du modèle, au format de son entraînement (`projet_14.mediqal`), en interrogeant
vLLM directement. Les requêtes partent une par une, pour limiter l'effet du regroupement des requêtes sur les réponses.

    uv run python scripts/evaluate.py --vllm-url http://localhost:8001 --label sft

Sorties : `reports/evaluation/<label>_answers.csv` (une ligne par question) et `<label>_summary.json`.
"""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd

from projet_14.mediqal import LETTERS, SYSTEM_PROMPT, answer_letters, parse_user_content, user_content

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "anonymized"
OUT = ROOT / "reports" / "evaluation"


def build_messages(question: dict) -> list[dict]:
    # Même prompt qu'à l'entraînement : cas, question, propositions, consigne
    row = {"clinical_case": question.get("cas_clinique"), "question": question["question"]}
    row |= {f"answer_{letter.lower()}": question["propositions"].get(letter, "") for letter in LETTERS}
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content(row, multiple_answers=question["reponses_multiples"])}]


def valid_format(letters: list[str], question: dict) -> bool:
    # Réponse exploitable : au moins une lettre, seulement des lettres proposées, en nombre conforme à la consigne
    count_ok = len(letters) >= 2 if question["reponses_multiples"] else len(letters) == 1
    return bool(letters) and set(letters) <= set(question["propositions"]) and count_ok


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
    payload = {"model": "qwen3-medical", "messages": build_messages(question), "temperature": 0,
               "max_tokens": 192 if question["reponses_multiples"] else 64}
    response = await client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    letters = sorted(answer_letters(text))
    return {
        "id": question["id"], "jeu": question["jeu"], "attendu": "".join(question["attendu"]),
        "lettres": "".join(letters), "juste": letters == question["attendu"],
        "dangereuse": bool(set(letters) & set(question["dangereuses"])),
        "format_valide": valid_format(letters, question), "reponse": text,
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
