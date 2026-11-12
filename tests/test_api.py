"""Tests de l'API, sans GPU : l'appel à vLLM est remplacé par une fausse réponse."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from projet_14.mediqal import MULTIPLE_ANSWERS, OPTIONS_HEADER, SINGLE_ANSWER

API_KEY = "cle-de-test"
QCM = {
    "cas_clinique": "Un homme de 45 ans présente des adénopathies cervicales bilatérales.",
    "question": "Le stade pathologique (PS) de ce patient est :",
    "propositions": {"A": "PS IIE", "B": "PS IIB", "C": "PS IIIA", "D": "PS IIIB", "E": "PS IVB"},
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return TestClient(main.app)


@pytest.fixture
def fake_vllm(monkeypatch):
    # Remplace vLLM : enregistre les messages reçus et renvoie la réponse choisie par le test
    calls = []

    def install(answer: str):
        async def call_vllm(messages, max_tokens):
            calls.append({"messages": messages, "max_tokens": max_tokens})
            return answer

        monkeypatch.setattr(main, "call_vllm", call_vllm)
        return calls

    return install


def audit_lines(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_cle_absente_ou_invalide(client):
    assert client.post("/qcm", json=QCM).status_code == 401
    assert client.post("/qcm", json=QCM, headers={"X-API-Key": "mauvaise"}).status_code == 401


def test_requete_invalide(client):
    headers = {"X-API-Key": API_KEY}
    assert client.post("/qcm", json=QCM | {"propositions": {"A": "seule"}}, headers=headers).status_code == 422
    assert client.post("/qcm", json=QCM | {"propositions": {"F": "x", "A": "y"}}, headers=headers).status_code == 422
    assert client.post("/qcm", json=QCM | {"question": ""}, headers=headers).status_code == 422


def test_reponse_unique(client, fake_vllm, tmp_path):
    calls = fake_vllm("D. PS IIIB")
    response = client.post("/qcm", json=QCM, headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    body = response.json()
    assert body["lettres"] == ["D"]
    assert body["reponse"] == "D. PS IIIB"
    assert body["avertissement"]

    # Prompt au format de l'entraînement : cas, question, propositions, consigne
    user = calls[0]["messages"][1]["content"]
    assert user.startswith(QCM["cas_clinique"])
    assert f"{OPTIONS_HEADER}\nA. PS IIE" in user
    assert user.endswith(SINGLE_ANSWER)


def test_reponses_multiples(client, fake_vllm):
    calls = fake_vllm("- A. PS IIE\n- C. PS IIIA")
    response = client.post("/qcm", json=QCM | {"reponses_multiples": True}, headers={"X-API-Key": API_KEY})

    assert response.json()["lettres"] == ["A", "C"]
    assert calls[0]["messages"][1]["content"].endswith(MULTIPLE_ANSWERS)
    assert calls[0]["max_tokens"] > 64  # une liste est plus longue qu'une réponse unique


def test_sans_cas_clinique(client, fake_vllm):
    calls = fake_vllm("B. PS IIB")
    client.post("/qcm", json=QCM | {"cas_clinique": None}, headers={"X-API-Key": API_KEY})
    assert calls[0]["messages"][1]["content"].startswith(QCM["question"])


def test_journal_audit_sans_texte_patient(client, fake_vllm, tmp_path):
    fake_vllm("D. PS IIIB")
    response = client.post("/qcm", json=QCM, headers={"X-API-Key": API_KEY})

    [record] = audit_lines(tmp_path)
    assert record["request_id"] == response.json()["request_id"]
    assert record["status"] == "ok"
    assert record["letters"] == ["D"]
    assert len(record["input_sha256"]) == 64
    assert "adénopathies" not in json.dumps(record, ensure_ascii=False)  # le cas n'est pas conservé


def test_vllm_indisponible(client, monkeypatch, tmp_path):
    async def call_vllm(messages, max_tokens):
        raise httpx.ConnectError("vLLM arrêté")

    monkeypatch.setattr(main, "call_vllm", call_vllm)
    response = client.post("/qcm", json=QCM, headers={"X-API-Key": API_KEY})

    assert response.status_code == 503
    assert audit_lines(tmp_path)[0]["status"] == "error"


def test_health_sans_vllm(client, monkeypatch):
    monkeypatch.setenv("VLLM_URL", "http://127.0.0.1:9")  # aucun serveur à cette adresse
    assert client.get("/health").json() == {"api": "ok", "vllm": "indisponible"}
