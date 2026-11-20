"""Évaluation externe du niveau de triage : 60 cas rédigés à la main, d'un jeu public indépendant du projet.

Jeu : `BenoitJT-GIRARD/clinical-triage-bilingual` (Hugging Face, CC BY 4.0), sous-ensemble `clinical_eval` :
20 urgences vitales, 20 modérées, 20 différées, en français et en anglais, dont des cas piégeux (faussement
rassurants ou alarmants, négations, constantes discordantes). Les niveaux ont été fixés par l'auteur du jeu, sans
validation par un médecin urgentiste : c'est une mesure indicative, pas une validation clinique.

Le modèle reçoit le prompt de l'API (`api/triage.py`), avec la description du patient comme cas clinique. Les
garde-fous sur les constantes ne s'appliquent pas (constantes non structurées) : seul le modèle est mesuré, avec et
sans la calibration de son a priori.

    uv run --with-requirements api/requirements.txt python scripts/evaluate_triage.py --vllm-url http://<vLLM>:8000

Sortie : `reports/evaluation/triage_externe.csv` (une ligne par cas).
"""

import argparse
import math
import sys
from pathlib import Path

import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "src")]
import triage as tri  # noqa: E402
from projet_14.mediqal import SYSTEM_PROMPT, user_content  # noqa: E402

DATASET = "BenoitJT-GIRARD/clinical-triage-bilingual"
LEVELS = {"URGENCE_VITALE": "urgence maximale", "URGENCE_MODEREE": "urgence modérée",
          "CONSULTATION_DIFFEREE": "urgence différée"}


def probabilities(client: httpx.Client, case: str) -> dict[str, float] | None:
    row = tri.prompt_row(None) | {"clinical_case": case}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content(row, multiple_answers=False)}]
    response = client.post("/v1/chat/completions", json={
        "model": "qwen3-medical", "messages": messages, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 20})
    response.raise_for_status()
    return tri.level_probabilities(response.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"])


def summary(levels: pd.Series, expected: pd.Series) -> str:
    gap = levels.map(tri.SEVERITY) - expected.map(tri.SEVERITY)  # > 0 : niveau moins grave qu'attendu
    vital_under = ((gap > 0) & (expected == "urgence maximale")).sum()
    return (f"justes {(gap == 0).sum()}/{len(gap)} ; sous-triage {(gap > 0).sum()} "
            f"(urgences vitales : {vital_under}/{(expected == 'urgence maximale').sum()}) ; sur-triage {(gap < 0).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-url", required=True, help="serveur vLLM (API compatible OpenAI)")
    args = parser.parse_args()

    cases = pd.read_json(hf_hub_download(DATASET, "clinical_eval.jsonl", repo_type="dataset"), lines=True)
    with httpx.Client(base_url=args.vllm_url, timeout=120) as client:
        prior = probabilities(client, tri.NEUTRAL_CASE)
        rows = []
        for case in cases.itertuples():
            p = probabilities(client, case.description)
            raw, calibrated = tri.model_levels(p, prior) if p else (None, None)
            rows.append({"id": case.id, "lang": case.lang, "type": case.case_type or "", "attendu": LEVELS[case.level],
                         "brut": raw, "calibre": calibrated, "p_maximale": p and round(p["urgence maximale"], 3)})
    results = pd.DataFrame(rows)

    print(f"a priori (cas vide) : { {level: round(v, 2) for level, v in prior.items()} }")
    for column in ("brut", "calibre"):
        print(f"{column:8} {summary(results[column], results['attendu'])}")
        print(pd.crosstab(results["attendu"], results[column]).to_string(), "\n")
    print("P(urgence maximale) brute, par niveau attendu (médiane) :",
          results.groupby("attendu")["p_maximale"].median().round(2).to_dict())

    out = ROOT / "reports" / "evaluation" / "triage_externe.csv"
    results.to_csv(out, index=False)
    print(f"Résultats : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
