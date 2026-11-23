"""Interface web du Space (`api/web_ui.py`) : mot de passe, redirection, évaluation dans le processus de l'API."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

gr = pytest.importorskip("gradio")  # Gradio n'est installé que pour l'image du Space et la CI

import triage as tri  # noqa: E402
import web_ui  # noqa: E402

RHUME = [35, "F", "rhume", "nez qui coule depuis 3 jours", "", 72, 120, None, 99, None, 37.0, None, None]


def response(level: str = "urgence différée") -> tri.TriageResponse:
    return tri.TriageResponse(
        request_id="r1", niveau=level, source="motif bénin", niveau_modele="urgence modérée", signes_de_gravite=[],
        justification="Motif bénin.", conduite=tri.CONDUITE[level], questions_complementaires=["Q ?"],
        modele="qwen3-medical", latence_ms=40, avertissement="Démonstration technique.")


@pytest.fixture
def client():
    async def fake_triage(request):
        return response()

    async def fake_health():
        return {"api": "ok", "vllm": "ok"}

    app = web_ui.mount(FastAPI(), fake_triage, fake_health, cases=[], user="jury", password="secret")
    return TestClient(app)


def test_racine_redirige_vers_l_interface(client):
    page = client.get("/", follow_redirects=False)
    assert (page.status_code, page.headers["location"]) == (307, "/ui")


def test_mot_de_passe_exige(client):
    # Sans session, l'interface ne livre rien d'utilisable ; le mot de passe protège l'endpoint payant
    assert client.get("/ui/config").status_code == 401
    assert client.post("/ui/login", data={"username": "jury", "password": "faux"}).status_code == 400
    assert client.post("/ui/login", data={"username": "jury", "password": "secret"}).status_code == 200
    assert client.get("/ui/config").status_code == 200


def test_requete_construite_sans_champs_vides():
    request = web_ui.build_request(*RHUME)
    assert request == {"age": 35, "sexe": "F", "motif": "rhume", "symptomes": "nez qui coule depuis 3 jours",
                       "constantes": {"fc": 72, "pas": 120, "spo2": 99, "temperature": 37.0}}


def test_evaluation_dans_le_processus():
    received = []

    async def fake_triage(request):
        received.append(request)
        return response()

    markdown, body = asyncio.run(web_ui.in_process(fake_triage)(*RHUME))
    assert received[0].motif == "rhume" and received[0].constantes.spo2 == 99
    assert "Urgence différée" in markdown and body["source"] == "motif bénin"


def test_evaluation_refusee_par_la_validation():
    async def fake_triage(request):
        raise AssertionError("le modèle ne doit pas être appelé")

    markdown, _ = asyncio.run(web_ui.in_process(fake_triage)(*([None] + RHUME[1:])))  # âge manquant
    assert markdown.startswith("❌ Requête refusée") and "`age`" in markdown


def test_serveur_d_inference_indisponible():
    async def fake_triage(request):
        raise HTTPException(status_code=503, detail="Serveur d'inférence indisponible")

    markdown, _ = asyncio.run(web_ui.in_process(fake_triage)(*RHUME))
    assert markdown == "❌ Serveur d'inférence indisponible"
