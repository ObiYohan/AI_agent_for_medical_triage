# Agent IA de triage médical : POC

## Sources des données et licences

| Source | Contenu utilisé | Langue | Licence | Usage dans le projet |
|---|---|---|---|---|
| [ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | QCM médicaux rédigés par des enseignants en médecine, avec ou sans cas clinique : configurations `mcqu` (une bonne réponse) et `mcqm` (plusieurs bonnes réponses), avec leur corrigé | fr | CC-BY-4.0 | SFT (`mcqu`, `mcqm`) et DPO (`mcqm` : `rejected` construit à partir des distracteurs et du corrigé) |
| [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | Questions biomédicales (examens, ChatDoctor, MedQuad, Wikipédia…) et réponses de plusieurs LLM notées par GPT-4 ; split `test` revu par des experts (Medical RewardBench) | en | MIT | SFT (meilleure réponse vérifiée) et DPO (paires vérifiées par le corrigé ou par l'écart de note) |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Modèle de base | — | Apache 2.0 | Point de départ du SFT LoRA puis du DPO |

- **Préparation** : un notebook par jeu dans `notebook/` (`data_prep_MediQAl`, `data_prep_SFT_MediQAl_mcqm`, `data_prep_DPO_MediQAl_mcqm`, `data_prep_SFT_UltraMedical`, `data_prep_DPO_UltraMedical`), avec les filtres, les contrôles et les limites de chaque source.
- **Données dérivées** : nettoyées, filtrées et anonymisées (Presidio), au format JSONL (`messages` pour le SFT, `prompt` / `chosen` / `rejected` pour le DPO). Elles ne sont pas versionnées dans git (`data/` est ignoré) et se régénèrent avec les notebooks.
- **Obligations** : CC-BY-4.0 et MIT imposent de citer la source et sa licence, ce que fait ce tableau et la fiche du modèle publié.
- **Modèle publié** : [`0biyohan/qwen3-1.7b-medical-qcm-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-qcm-fr) (dépôt privé, Apache 2.0).

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

Prérequis : Docker avec accès au GPU (NVIDIA Container Toolkit) et un jeton Hugging Face en lecture. Le modèle
est téléchargé depuis le dépôt privé [`0biyohan/qwen3-1.7b-medical-qcm-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-qcm-fr),
à une révision fixée dans `docker-compose.yml` (variables `MODEL` et `MODEL_REVISION` pour en changer).

```bash
cp .env.example .env          # puis définir API_KEY et HF_TOKEN
docker compose up -d          # premier lancement : image vLLM (~10 Go) et modèle (3,4 Go), mis en cache ensuite
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
| `tests/test_api.py` | tests de l'API sans GPU (vLLM simulé), lancés aussi par la CI : `uv run --with-requirements api/requirements.txt --with pytest pytest tests` |
| `tests/integration/` | robustesse et audit de traçabilité contre la pile démarrée : `INTEGRATION_API_URL=http://localhost:8080 API_KEY=<clé> uv run --with-requirements api/requirements.txt --with pytest pytest tests/integration` (arrêt / redémarrage de vLLM avec `ROBUSTESSE_ARRET_VLLM=1`) |
| `scripts/latency.py` | latence et débit à 1, 10 et 50 requêtes simultanées (API de bout en bout, puis vLLM seul : TTFT, TPOT), exactitude des réponses ; résultats dans MLflow (expérience `qwen3-1.7b-serving`) : `API_KEY=<clé> uv run python scripts/latency.py [--mlflow dagshub]` |
| `.github/workflows/ci.yml` | CI/CD : à chaque push, tests puis construction de l'image de l'API ; sur `master`, publication sur GHCR (`ghcr.io/obiyohan/triage-api:<commit>`), déploiement sur la machine GPU (runner auto-hébergé, étiquette `gpu`, dossier `~/p14-deploy`) et tests d'intégration contre la pile déployée |

**Déploiement continu** : secrets `API_KEY` et `HF_TOKEN` dans GitHub (*Settings → Secrets and variables → Actions*),
runner auto-hébergé avec l'étiquette `gpu` sur la machine qui sert la démonstration. Un nouveau commit sur `master`
remplace le conteneur de l'API sans redémarrer vLLM. Retour à une version précédente : relancer (*Re-run all jobs*)
le workflow d'un commit antérieur dans l'onglet *Actions*.

Le port de vLLM n'est pas publié : seule l'API est accessible. Démonstration technique, ne remplace pas un avis médical.
