# API de démonstration sur un Space Hugging Face (Docker, CPU) ; l'inférence est faite par un Inference Endpoint.
# Fichiers assemblés par scripts/deploy_space.py : main.py, triage.py, web_ui.py (interface /ui), triage_cases.json,
# requirements.txt, requirements-ui.txt (Gradio), projet_14/ (module partagé du prompt).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt

COPY projet_14/ projet_14/
COPY main.py triage.py web_ui.py triage_cases.json ./

# Un Space tourne avec l'utilisateur 1000 : journal d'audit dans /tmp (seul dossier inscriptible, effacé au
# redémarrage). Chaque ligne est aussi écrite sur la sortie standard, visible dans l'onglet Logs du Space.
ENV AUDIT_LOG=/tmp/audit/audit.jsonl
# Interface /ui (Gradio) : fichiers temporaires dans /tmp, pas de télémétrie
ENV GRADIO_TEMP_DIR=/tmp/gradio HOME=/tmp GRADIO_ANALYTICS_ENABLED=False

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
