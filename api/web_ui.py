"""Interface web (Gradio) du triage : formulaire patient, cas de démonstration, niveau de priorité expliqué.

Une seule interface, servie de deux façons :

- dans le Space (démo cloud), sous `/ui`, protégée par un identifiant et un mot de passe (`mount`) : elle appelle le
  triage dans le processus de l'API, sans passer par la clé d'API, et chaque évaluation est tracée dans le journal
  d'audit comme une requête `/triage` ;
- sur le poste du développeur (`scripts/ui.py`), avec le choix de la cible (pile locale ou cloud), par HTTP.

Le mot de passe protège l'endpoint payant : chaque évaluation peut réveiller l'Inference Endpoint.
"""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

import triage as tri

CONSTANTES = ("fc", "pas", "pad", "spo2", "fr", "temperature", "eva")
INTEGERS = ("fc", "pas", "pad", "spo2", "fr", "eva")  # entiers attendus par l'API
NIVEAU_COULEUR = {"urgence maximale": "#c62828", "urgence modérée": "#ef6c00", "urgence différée": "#2e7d32"}
WARNING = "Démonstration technique (POC) : ne remplace pas l'avis d'un soignant."


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def case_label(case: dict) -> str:
    return f"{case['id']} · {case['motif']} (attendu : {case['attendu']})"


def case_fields(cases: list[dict], label: str) -> list:
    # Valeurs du formulaire pour un cas de démonstration, dans l'ordre des champs
    case = next(c for c in cases if case_label(c) == label)
    constantes = case.get("constantes") or {}
    return ([case["age"], case.get("sexe"), case["motif"], case["symptomes"], case.get("antecedents") or ""]
            + [constantes.get(name) for name in CONSTANTES] + [constantes.get("conscience")])


def build_request(age, sexe, motif, symptomes, antecedents, *values) -> dict:
    # Corps de la requête /triage : les champs vides ne sont pas envoyés
    *numbers, conscience = values
    constantes = {name: value for name, value in zip(CONSTANTES, numbers) if value is not None}
    for name in INTEGERS:
        if name in constantes:
            constantes[name] = int(constantes[name])
    if conscience:
        constantes["conscience"] = conscience
    request = {"age": int(age) if age is not None else None, "motif": motif or "", "symptomes": symptomes or "",
               "constantes": constantes}
    if sexe:
        request["sexe"] = sexe
    if antecedents and antecedents.strip():
        request["antecedents"] = antecedents
    return request


def refusal(errors: list[tuple[tuple, str]]) -> str:
    # Champs refusés par la validation : (chemin du champ, message)
    return "❌ Requête refusée\n\n" + "\n".join(f"- `{'.'.join(map(str, loc))}` : {msg}" for loc, msg in errors)


def render(body: dict) -> str:
    color = NIVEAU_COULEUR.get(body["niveau"], "#555")
    lines = [
        f"<div style='background:{color};color:white;padding:12px 16px;border-radius:8px;font-size:1.4em;"
        f"font-weight:bold'>{body['niveau'].capitalize()}</div>",
        "",
        f"**Décidé par** : {body['source']} · **avis du modèle** : {body['niveau_modele'] or '—'}"
        + (f" (avant calibration : {body['niveau_modele_brut']})"
           if body.get("niveau_modele_brut") not in (None, body["niveau_modele"]) else "") + " · "
        f"**latence** : {body['latence_ms']} ms",
        "",
        "**Signes de gravité (constantes hors seuils)** : " + ("; ".join(body["signes_de_gravite"]) or "aucun"),
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


def build(evaluate: Callable, check_health: Callable, cases: list[dict], targets: dict[str, str] | None = None,
          subtitle: str = "") -> gr.Blocks:
    """Interface Gradio. Avec `targets`, une cible est choisie dans la page, et `evaluate` et `check_health`
    reçoivent (cible, autre URL) avant les champs du formulaire."""
    with gr.Blocks(title="Triage aux urgences : démonstration") as demo:
        gr.Markdown(f"## Agent IA de triage aux urgences\n{subtitle}\n\n{WARNING}")
        target_inputs = []
        with gr.Row():
            if targets:
                target_inputs = [gr.Radio(list(targets), value=next(iter(targets)), label="Cible"),
                                 gr.Textbox(label="Autre URL (facultatif)", placeholder="https://…", scale=2)]
            check = gr.Button("État du service", scale=0)
        status = gr.Markdown()

        case = gr.Dropdown([case_label(c) for c in cases], label="Cas de démonstration (remplit le formulaire)")
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
        check.click(check_health, target_inputs, status)
        case.change(lambda label: case_fields(cases, label), case, fields)
        run.click(evaluate, [*target_inputs, *fields], [result, raw])
    return demo


def in_process(triage_fn: Callable[[tri.TriageRequest], Awaitable[tri.TriageResponse]]) -> Callable:
    """Évaluation dans le processus de l'API (Space) : même validation et même traitement que POST /triage."""

    async def evaluate(*fields):
        request = build_request(*fields)
        try:
            body = (await triage_fn(tri.TriageRequest(**request))).model_dump()
        except ValidationError as error:
            return refusal([(e["loc"], e["msg"]) for e in error.errors()]), request
        except HTTPException as error:  # entrée trop longue (422), serveur d'inférence indisponible (503)
            return f"❌ {error.detail}", request
        return render(body), body

    return evaluate


def mount(app: FastAPI, triage_fn: Callable[[tri.TriageRequest], Awaitable[tri.TriageResponse]],
          health_fn: Callable[[], Awaitable[dict]], cases: list[dict], user: str, password: str) -> FastAPI:
    """Sert l'interface sous /ui, derrière un identifiant et un mot de passe ; / redirige vers /ui."""

    async def check_health():
        state = await health_fn()
        mark = "✅" if all(value == "ok" for value in state.values()) else "⚠️"
        return f"{mark} " + ", ".join(f"{key} = {value}" for key, value in state.items())

    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse("/ui")

    demo = build(in_process(triage_fn), check_health, cases, subtitle="Démonstration cloud : modèle servi par vLLM (GPU T4).")
    return gr.mount_gradio_app(app, demo, path="/ui", auth=(user, password), ssr_mode=False,
                               auth_message="Démonstration réservée : identifiant et mot de passe communiqués à part.")
