"""Interface web de test de l'endpoint de triage (/triage), en local comme sur le cloud.

    uv run --with gradio python scripts/ui.py                 puis ouvrir http://127.0.0.1:7860

Le choix de la cible (pile locale, Space Hugging Face ou autre URL) se fait dans la page. Les secrets restent côté
serveur Gradio, jamais dans la page : clé de l'API lue dans API_KEY ou dans .env, jeton Hugging Face (Space privé) dans
HF_TOKEN ou dans la connexion locale (`hf auth login`). L'interface n'écoute que sur la machine (127.0.0.1) : pas de
lien public, puisqu'elle appelle l'API avec la clé.
"""

import json
import os
from pathlib import Path

import gradio as gr
import httpx
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "scripts" / "triage_cases.json").read_text(encoding="utf-8"))
TARGETS = {
    "Local (docker compose)": "http://localhost:8080",
    "Cloud (Space Hugging Face)": "https://0biyohan-triage-api.hf.space",
}
# Réponse : jusqu'à 300 s (réveil de l'Inference Endpoint) ; connexion : 10 s (URL fausse ou pile arrêtée)
TIMEOUT = httpx.Timeout(300, connect=10)
CONSTANTES = ("fc", "pas", "pad", "spo2", "fr", "temperature", "eva")
NIVEAU_COULEUR = {"urgence maximale": "#c62828", "urgence modérée": "#ef6c00", "urgence différée": "#2e7d32"}


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


def load_case(label: str):
    case = next(c for c in CASES if case_label(c) == label)
    constantes = case.get("constantes", {})
    return ([case["age"], case.get("sexe"), case["motif"], case["symptomes"], case.get("antecedents") or ""]
            + [constantes.get(name) for name in CONSTANTES] + [constantes.get("conscience")])


def case_label(case: dict) -> str:
    return f"{case['id']} · {case['motif']} (attendu : {case['attendu']})"


def build_request(age, sexe, motif, symptomes, antecedents, *values) -> dict:
    *numbers, conscience = values
    constantes = {name: value for name, value in zip(CONSTANTES, numbers) if value is not None}
    for name in ("fc", "pas", "pad", "spo2", "fr", "eva"):  # entiers attendus par l'API
        if name in constantes:
            constantes[name] = int(constantes[name])
    if conscience:
        constantes["conscience"] = conscience
    request = {"age": int(age) if age is not None else None, "motif": motif, "symptomes": symptomes,
               "constantes": constantes}
    if sexe:
        request["sexe"] = sexe
    if antecedents.strip():
        request["antecedents"] = antecedents
    return request


def render(body: dict) -> str:
    color = NIVEAU_COULEUR.get(body["niveau"], "#555")
    lines = [
        f"<div style='background:{color};color:white;padding:12px 16px;border-radius:8px;font-size:1.4em;"
        f"font-weight:bold'>{body['niveau'].capitalize()}</div>",
        "",
        f"**Décidé par** : {body['source']} · **avis du modèle** : {body['niveau_modele'] or '—'} · "
        f"**latence** : {body['latence_ms']} ms",
        "",
        "**Signes de gravité** : " + ("; ".join(body["signes_de_gravite"]) or "aucun"),
        "",
        f"**Justification** : {body['justification']}",
        "",
        f"**Conduite à tenir** : {body['conduite']}",
        "",
        "**Questions complémentaires** :",
        *[f"- {question}" for question in body["questions_complementaires"]],
        "",
        f"<small>Modèle `{body['modele']}` · requête `{body['request_id']}` (journal d'audit)</small>",
        "",
        f"> {body['avertissement']}",
    ]
    return "\n".join(lines)


def triage(target, custom, *fields):
    url = target_url(target, custom)
    request = build_request(*fields)
    try:
        response = httpx.post(f"{url}/triage", json=request, headers=headers(url), timeout=TIMEOUT)
    except httpx.HTTPError as error:
        return f"❌ `{url}` injoignable : {type(error).__name__}", request
    try:
        body = response.json()
    except ValueError:
        body = {"texte": response.text[:500]}
    if response.status_code == 200:
        return render(body), body
    if response.status_code == 422:  # champs refusés par la validation : on les liste
        errors = body.get("detail", [])
        details = "\n".join(f"- `{'.'.join(str(p) for p in e.get('loc', [])[1:])}` : {e.get('msg')}"
                            for e in errors) if isinstance(errors, list) else f"- {errors}"
        return f"❌ Requête refusée (422)\n\n{details}", body
    return f"❌ HTTP {response.status_code} : {body.get('detail', body)}", body


with gr.Blocks(title="Triage : test de l'API") as demo:
    gr.Markdown("## Agent IA de triage : test de l'endpoint `/triage`\n"
                "Démonstration technique (POC) : ne remplace pas l'avis d'un soignant.")
    with gr.Row():
        target = gr.Radio(list(TARGETS), value=next(iter(TARGETS)), label="Cible")
        custom = gr.Textbox(label="Autre URL (facultatif)", placeholder="https://…", scale=2)
        check = gr.Button("Vérifier /health", scale=0)
    status = gr.Markdown()

    case = gr.Dropdown([case_label(c) for c in CASES], label="Cas de démonstration (remplit le formulaire)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                age = gr.Number(label="Âge", value=58, precision=0, minimum=0, maximum=120)
                sexe = gr.Radio(["F", "M"], label="Sexe", value="M")
            motif = gr.Textbox(label="Motif de recours", value="douleur thoracique")
            symptomes = gr.Textbox(label="Symptômes", lines=4, value="douleur constrictive depuis 40 minutes, "
                                   "irradiant dans le bras gauche, avec des sueurs")
            antecedents = gr.Textbox(label="Antécédents", value="tabagisme")
            gr.Markdown("**Constantes** (laisser vide si non mesurée)")
            with gr.Row():
                fc = gr.Number(label="FC (/min)", value=105, precision=0)
                pas = gr.Number(label="PAS (mmHg)", value=150, precision=0)
                pad = gr.Number(label="PAD (mmHg)", value=90, precision=0)
                spo2 = gr.Number(label="SpO₂ (%)", value=96, precision=0)
            with gr.Row():
                fr = gr.Number(label="FR (/min)", precision=0)
                temperature = gr.Number(label="Température (°C)")
                eva = gr.Number(label="Douleur (0-10)", precision=0)
                conscience = gr.Dropdown(["normale", "alteree"], label="Conscience", value=None)
            run = gr.Button("Évaluer le niveau de priorité", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            with gr.Accordion("Requête ou réponse brute (JSON)", open=False):
                raw = gr.JSON()

    fields = [age, sexe, motif, symptomes, antecedents, fc, pas, pad, spo2, fr, temperature, eva, conscience]
    check.click(health, [target, custom], status)
    case.change(load_case, case, fields)
    run.click(triage, [target, custom, *fields], [result, raw])


if __name__ == "__main__":
    if not api_key():
        print("Attention : pas de clé d'API (variable API_KEY ou ligne API_KEY= dans .env) ; l'API répondra 401.")
    demo.launch(server_name="127.0.0.1")
