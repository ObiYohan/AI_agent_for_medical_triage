"""Mesures de latence de la démonstration (API FastAPI + vLLM), enregistrées dans MLflow.

1. **Bout en bout, à travers l'API** : des QCM du jeu de test (`mcqu` et `mcqm` en alternance) sont rejoués
   contre `POST /qcm` à plusieurs niveaux de concurrence (requêtes simultanées). Pour chaque niveau : latence
   p50 / p95 / max, débit (requêtes par seconde), erreurs, et exactitude des réponses par rapport au corrigé.
2. **vLLM seul** (`vllm bench serve`, dans le conteneur) : TTFT (délai avant le premier token), TPOT (temps par
   token généré) et débit en tokens par seconde, sur des prompts synthétiques de la taille des nôtres.

Usage, pile démarrée (docker compose up -d) :

    API_KEY=<clé> uv run python scripts/latency.py                       # MLflow local (../mlflow.db)
    API_KEY=<clé> uv run python scripts/latency.py --mlflow dagshub      # MLflow sur DagsHub
    API_KEY=<clé> uv run python scripts/latency.py --no-vllm-bench       # sans la mesure vLLM seule
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import mlflow
import pandas as pd

from projet_14.mediqal import answer_letters, parse_user_content

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "anonymized"
EXPERIMENT = "qwen3-1.7b-serving"
DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME = "obiyohan", "my-ml-mlflow"
# Taille des prompts synthétiques de vllm bench : proche de nos QCM (~800 caractères, ~300 tokens ; réponses courtes)
BENCH_INPUT_TOKENS, BENCH_OUTPUT_TOKENS = 300, 32
MODEL_REPO = "0biyohan/qwen3-1.7b-medical-qcm-fr"  # tokenizer des prompts synthétiques (déjà en cache dans vLLM)


def load_qcms(n: int) -> list[dict]:
    # QCM du jeu de test, avec leur corrigé : mcqu et mcqm en alternance
    sources = {}
    for stem in ("sft_mediqal_mcqu", "sft_mediqal_mcqm"):
        with (DATA / f"{stem}_test.anonymized.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        sources[stem] = [
            parse_user_content(r["messages"][1]["content"]) | {
                "source": stem, "expected": sorted(answer_letters(r["messages"][2]["content"]))}
            for r in records
        ]
    mixed = [qcm for pair in zip(*sources.values()) for qcm in pair]
    return mixed[:n]


async def run_level(client: httpx.AsyncClient, qcms: list[dict], concurrency: int) -> tuple[list[dict], float]:
    # Toutes les requêtes du niveau, au plus `concurrency` à la fois ; renvoie le détail et la durée totale
    semaphore = asyncio.Semaphore(concurrency)

    async def one(qcm: dict) -> dict:
        body = {key: qcm[key] for key in ("cas_clinique", "question", "propositions", "reponses_multiples")}
        async with semaphore:
            start = time.perf_counter()
            response = await client.post("/qcm", json=body)
            latency_ms = (time.perf_counter() - start) * 1000
        row = {"concurrency": concurrency, "source": qcm["source"], "status": response.status_code,
               "latency_ms": latency_ms}
        if response.status_code == 200:
            answer = response.json()
            row |= {"server_latency_ms": answer["latence_ms"], "letters": answer["lettres"],
                    "format_valide": answer["format_valide"], "correct": answer["lettres"] == qcm["expected"]}
        return row

    start = time.perf_counter()
    rows = await asyncio.gather(*(one(qcm) for qcm in qcms))
    return rows, time.perf_counter() - start


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1] if len(values) > 1 else values[0]


async def measure_api(url: str, api_key: str, qcms: list[dict], levels: list[int]) -> tuple[pd.DataFrame, dict]:
    metrics, all_rows = {}, []
    async with httpx.AsyncClient(base_url=url, headers={"X-API-Key": api_key}, timeout=300) as client:
        await run_level(client, qcms[:5], 1)  # échauffement : premières compilations, caches
        for concurrency in levels:
            rows, duration = await run_level(client, qcms, concurrency)
            all_rows += rows
            ok = [r for r in rows if r["status"] == 200]
            latencies = [r["latency_ms"] for r in ok]
            metrics |= {
                f"api_latency_p50_ms_c{concurrency}": percentile(latencies, 50),
                f"api_latency_p95_ms_c{concurrency}": percentile(latencies, 95),
                f"api_latency_max_ms_c{concurrency}": max(latencies),
                f"api_server_latency_p50_ms_c{concurrency}": percentile([r["server_latency_ms"] for r in ok], 50),
                f"api_throughput_rps_c{concurrency}": len(ok) / duration,
                f"api_errors_c{concurrency}": len(rows) - len(ok),
            }
            print(f"concurrence {concurrency:>3} : p50 {metrics[f'api_latency_p50_ms_c{concurrency}']:7.0f} ms, "
                  f"p95 {metrics[f'api_latency_p95_ms_c{concurrency}']:7.0f} ms, "
                  f"{metrics[f'api_throughput_rps_c{concurrency}']:5.1f} req/s, {len(rows) - len(ok)} erreur(s)")

    details = pd.DataFrame(all_rows)
    first_level = details[(details["concurrency"] == levels[0]) & (details["status"] == 200)]
    for source, group in first_level.groupby("source"):
        metrics[f"accuracy_{source}"] = group["correct"].mean()
        metrics[f"format_valide_{source}"] = group["format_valide"].mean()
    return details, metrics


def measure_vllm(levels: list[int], num_prompts: int) -> dict:
    # vllm bench serve, exécuté dans le conteneur vLLM (son port n'est pas publié)
    metrics = {}
    for concurrency in levels:
        command = [
            "docker", "compose", "exec", "-T", "vllm", "vllm", "bench", "serve",
            "--backend", "openai-chat", "--base-url", "http://localhost:8000", "--endpoint", "/v1/chat/completions",
            "--model", "qwen3-medical", "--tokenizer", MODEL_REPO,
            "--dataset-name", "random", "--random-input-len", str(BENCH_INPUT_TOKENS),
            "--random-output-len", str(BENCH_OUTPUT_TOKENS), "--ignore-eos",
            "--num-prompts", str(num_prompts), "--max-concurrency", str(concurrency),
            "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,95",
            "--save-result", "--result-dir", "/tmp/bench", "--result-filename", "bench.json",
        ]
        subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
        cat = ["docker", "compose", "exec", "-T", "vllm", "cat", "/tmp/bench/bench.json"]
        result = json.loads(subprocess.run(cat, check=True, capture_output=True, cwd=ROOT).stdout)
        metrics |= {
            f"vllm_ttft_p50_ms_c{concurrency}": result["p50_ttft_ms"],
            f"vllm_ttft_p95_ms_c{concurrency}": result["p95_ttft_ms"],
            f"vllm_tpot_p50_ms_c{concurrency}": result["p50_tpot_ms"],
            f"vllm_e2e_p50_ms_c{concurrency}": result["p50_e2el_ms"],
            f"vllm_output_tokens_per_s_c{concurrency}": result["output_throughput"],
            f"vllm_requests_per_s_c{concurrency}": result["request_throughput"],
        }
        print(f"vLLM seul, concurrence {concurrency:>3} : TTFT p50 {result['p50_ttft_ms']:6.1f} ms, "
              f"TPOT p50 {result['p50_tpot_ms']:5.1f} ms, {result['output_throughput']:7.0f} tokens/s")
    return metrics


def serving_params() -> dict:
    # Configuration servie, relue dans docker-compose.yml : chaque mesure reste liée à sa configuration
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    params = {"vllm_image": re.search(r"image: (vllm/\S+)", compose).group(1)}
    for option in ("max-model-len", "gpu-memory-utilization"):
        params[option.replace("-", "_")] = re.search(rf"--{option}=(\S+)", compose).group(1)
    params["model"] = re.search(r"--model=\$\{MODEL:-(\S+)}", compose).group(1)
    params["model_revision"] = re.search(r"--revision=\$\{MODEL_REVISION:-(\S+)}", compose).group(1)
    try:
        gpu = subprocess.run(["docker", "compose", "exec", "-T", "vllm", "nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"], capture_output=True, text=True, cwd=ROOT, check=True)
        params["gpu"] = gpu.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        params["gpu"] = "inconnu"
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=100, help="requêtes par niveau de concurrence")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--mlflow", choices=["local", "dagshub"], default="local")
    parser.add_argument("--no-vllm-bench", action="store_true", help="ne pas mesurer vLLM seul")
    args = parser.parse_args()
    api_key = os.environ["API_KEY"]

    if args.mlflow == "dagshub":
        import dagshub

        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)

    params = serving_params() | {"api_url": args.url, "requests_per_level": args.n,
                                 "concurrency_levels": " ".join(map(str, args.concurrency))}
    qcms = load_qcms(args.n)
    with mlflow.start_run(run_name=f"latency-{time.strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params(params)
        details, metrics = asyncio.run(measure_api(args.url, api_key, qcms, args.concurrency))
        if not args.no_vllm_bench:
            metrics |= measure_vllm(args.concurrency, num_prompts=args.n)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            details.to_csv(Path(tmp) / "requests.csv", index=False)  # une ligne par requête, pour l'analyse
            (Path(tmp) / "summary.json").write_text(json.dumps(params | metrics, indent=2), encoding="utf-8")
            mlflow.log_artifacts(tmp)
        print(f"Run MLflow : {mlflow.active_run().info.run_id} (expérience {EXPERIMENT})")


if __name__ == "__main__":
    main()
