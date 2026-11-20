---
title: Agent de triage médical (API de démonstration)
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8080
pinned: false
short_description: API de triage médical, inférence vLLM sur HF Endpoint
---

# Agent IA de triage médical : API de démonstration

API FastAPI du POC d'agent de triage médical. À partir d'un patient décrit en langage naturel (âge, motif,
symptômes, antécédents, constantes), elle propose un niveau de priorité avec le modèle
[`0biyohan/qwen3-1.7b-medical-qcm-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-qcm-fr) (Qwen3-1.7B-Base,
SFT LoRA + DPO), servi par vLLM sur un Inference Endpoint Hugging Face.

| Point d'accès | Rôle |
|---|---|
| `GET /health` | état de l'API et du serveur d'inférence |
| `POST /triage` | niveau de priorité, signes de gravité, justification, conduite à tenir et questions complémentaires (clé obligatoire dans l'en-tête `X-API-Key`) |
| `GET /docs` | documentation interactive |

Chaque requête est tracée (journal d'audit sans texte patient, visible dans les logs du Space).

> **Démonstration technique uniquement** : ne remplace pas un avis médical.

Code source et documentation : [ObiYohan/AI_agent_for_medical_triage](https://github.com/ObiYohan/AI_agent_for_medical_triage).
Ce Space est mis à jour automatiquement par la CI/CD du dépôt (`scripts/deploy_space.py`).
