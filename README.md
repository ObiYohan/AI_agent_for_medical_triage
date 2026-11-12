# Agent IA de triage médical : POC

## Anonymisation RGPD

Le script `rgpd-anonymize` anonymise un dataset JSONL au format SFT (`messages`) ou DPO
(`prompt` / `chosen` / `rejected`) avec Presidio et spaCy (`fr_core_news_md`, `en_core_web_md`).

```bash
uv run rgpd-anonymize notebook/sft_mediqal_mcqu.jsonl --lang fr
```

| Sortie | Contenu | Versionner |
|---|---|---|
| `data/anonymized/<nom>.anonymized.jsonl` | Dataset anonymisé, avec un champ `anonymization` par enregistrement | Oui |
| `reports/rgpd/<nom>.report.md` / `.json` | Paramètres, règles, justification des exclusions, comptes, empreintes SHA-256 | Oui |
| `reports/rgpd/<nom>.audit.jsonl` | Une ligne par entité remplacée (champ, position, type, détecteur), sans valeur d'origine | Oui |
| `reports/rgpd/<nom>.review.txt` | Échantillon de contrôle manuel, avec valeurs d'origine | **Non** (ignoré par git) |

Options : `--threshold` (score minimal, 0.35 par défaut), `--review-sample` (taille de l'échantillon
de contrôle, 0 pour le désactiver), `--output-dir`, `--report-dir`, `--seed`. La langue de chaque
enregistrement est lue dans son champ `lang`, sinon `--lang` s'applique.

Les règles propres au domaine médical (éponymes, civilités, dates, entités conservées) sont
décrites dans le rapport généré.

## Démonstration locale (phase 3)

Le modèle fusionné (SFT + DPO, section 7.5 de `notebook/sft_lora.ipynb`) est servi par **vLLM** et exposé par une
API **FastAPI**. Les deux services tournent dans Docker (`docker-compose.yml`).

Prérequis : Docker avec accès au GPU (NVIDIA Container Toolkit) et le modèle dans
`outputs/dpo-lora-r16-v3-beta0.1-lr5e-05-seed42/merged` (autre dossier : variable `MODEL_DIR`).

```bash
cp .env.example .env          # puis définir API_KEY
docker compose up -d          # premier lancement : téléchargement de l'image vLLM (~10 Go) et chargement du modèle
curl localhost:8080/health
```

Requête :

```bash
curl -X POST localhost:8080/qcm -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{
  "cas_clinique": "Une femme de 30 ans consulte pour une fièvre à 39 °C et une toux productive depuis 3 jours.",
  "question": "Quel examen demandez-vous en première intention ?",
  "propositions": {"A": "Radiographie thoracique", "B": "Scanner cérébral", "C": "Échographie abdominale",
                   "D": "IRM lombaire", "E": "Électromyogramme"},
  "reponses_multiples": false
}'
```

| Élément | Rôle |
|---|---|
| `POST /qcm` | QCM au format de l'entraînement → lettre(s) et texte de la réponse. Clé d'API obligatoire (`X-API-Key`) |
| `GET /health` | état de l'API et de vLLM |
| `logs/audit.jsonl` | une ligne par requête : identifiant, horodatage, modèle, latence, lettres, empreinte SHA-256 de l'entrée (le texte du cas n'est pas conservé) |
| `tests/` | tests de l'API sans GPU (vLLM simulé) : `uv run --with-requirements api/requirements.txt --with pytest pytest tests` |
| `.github/workflows/ci.yml` | à chaque push : tests, puis construction de l'image de l'API |

Le port de vLLM n'est pas publié : seule l'API est accessible. Démonstration technique, ne remplace pas un avis médical.
