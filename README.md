# Agent IA de triage médical : POC

## Sources des données et licences

| Source | Contenu utilisé | Langue | Licence | Usage dans le projet |
|---|---|---|---|---|
| [ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | Cas cliniques rédigés par des enseignants en médecine, avec leurs questions : ouvertes à réponse rédigée (`oeq`), QCM à une bonne réponse (`mcqu`) ou à plusieurs (`mcqm`), avec leur corrigé | fr | CC-BY-4.0 | SFT (cas sélectionnés pour le triage) et DPO (QCM d'autres cas : `rejected` construite à partir du corrigé) |
| [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | Vignettes cliniques d'examen (MedQA, MedMCQA) et questions de patients (ChatDoctor), réponses de plusieurs LLM notées par GPT-4 ; split `test` revu par des experts (Medical RewardBench) | en | MIT | SFT (meilleure réponse vérifiée) et DPO (paires vérifiées ; test expert), sur des questions distinctes |
| [BenoitJT-GIRARD/clinical-triage-bilingual](https://huggingface.co/datasets/BenoitJT-GIRARD/clinical-triage-bilingual) | 60 cas de triage rédigés à la main (`clinical_eval`), avec un niveau de référence fixé par l'auteur du jeu, sans validation médicale | fr, en | CC-BY-4.0 | Évaluation externe du niveau de triage seulement (`scripts/evaluate_triage.py`) ; rien n'est redistribué |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Modèle de base | — | Apache 2.0 | Point de départ du SFT LoRA puis du DPO |

Jeux étudiés et non retenus : **MedQuAD** (questions générales sur les maladies, « What are the symptoms of X? », sans
patient décrit ni réponse rejetée ; une partie figure déjà dans UltraMedical) et **FrenchMedMCQA** (QCM de pharmacie
sans cas clinique). Ni l'un ni l'autre ne ressemble à l'entrée d'un triage.

- **Préparation** : un notebook par jeu dans `notebook/`, avec les filtres, les contrôles et les limites de chaque
  source. SFT en version 4 (`data/v4/`), DPO en version 5 (`data/v5/`) :

  | Notebook | Sortie | Contenu |
  |---|---|---|
  | `prep_SFT_MediQAl_oeq` | `data/v4/sft_mediqal_oeq_*` | 1 098 questions ouvertes, 204 cas |
  | `prep_SFT_MediQAl_mcqu` | `data/v4/sft_mediqal_mcqu_*` | 1 200 QCM à une bonne réponse, 293 cas |
  | `prep_SFT_MediQAl_mcqm` | `data/v4/sft_mediqal_mcqm_*` | 700 QCM à plusieurs bonnes réponses, 251 cas |
  | `prep_SFT_UltraMedical` | `data/v4/sft_ultramedical_*` | 2 000 questions en anglais |
  | `prep_DPO_UltraMedical` | `data/v5/dpo_ultramedical_*` | 800 paires (train + validation), réponses de 2 000 caractères au plus, et 756 paires de test revues par des experts |
  | `prep_DPO_MediQAl` | `data/v5/dpo_mediqal_*` | 2 156 paires de QCM (726 `mcqu`, 1 430 `mcqm`) : corrigé contre une erreur construite (moitié de substitutions, puis autant d'omissions que d'ajouts) |

  Ordre d'exécution : les trois notebooks SFT MediQAl (indépendants), puis `prep_DPO_MediQAl` (il écarte les cas
  du SFT, lus dans `data/v4/anonymized`) ; `prep_SFT_UltraMedical`, puis `prep_DPO_UltraMedical` (son contrôle final
  lit le SFT anglais).
- **Une ligne par cas clinique** (MediQAl) : le cas, puis ses questions dans l'ordre du récit, chacune suivie de sa
  réponse. À l'entraînement, toutes les réponses sont apprises (`assistant_only_loss`), chacune avec l'historique qui
  la précède. Format commun : `src/projet_14/conversation.py`.
- **Sélection pour le triage** (`src/projet_14/selection.py`), commune aux deux langues : patient décrit obligatoire,
  priorité aux cas avec constantes vitales et vocabulaire d'urgence, budget de questions réparti entre les splits,
  plafond par spécialité (ou par source). Environ 5 000 paires SFT : 2 998 en français, 2 000 en anglais. Aucune
  relecture clinique humaine dans le POC : la validation repose sur les corrigés, les notes et les contrôles automatiques.
- **Répartition** : train / validation / test (70 / 10 / 20) tirés de l'empreinte du cas clinique (ou de la question),
  identiques d'un notebook à l'autre. Le DPO MediQAl n'utilise que des cas non retenus pour le SFT ; pour
  UltraMedical, un premier tirage attribue chaque question au SFT ou au DPO.
- **Données dérivées** : nettoyées, filtrées et anonymisées (Presidio), au format JSONL (`messages` pour le SFT,
  `prompt` / `chosen` / `rejected` pour le DPO), lisible par `datasets`. Elles ne sont pas versionnées dans git
  (`data/` est ignoré) et se régénèrent avec les notebooks.
- **Publication** : dataset public [`0biyohan/triage-medical-bilingual`](https://huggingface.co/datasets/0biyohan/triage-medical-bilingual),
  une configuration par notebook, versionné par étiquette (`v1.0` : SFT v4 + DPO v5). Envoi des seuls fichiers
  anonymisés et de leurs rapports : `uv run python scripts/push_dataset.py --tag v1.0`. Fiche : `deploy/dataset/README.md` ;
  fiche du modèle : `deploy/model/README.md`.
- **Obligations** : CC-BY-4.0 et MIT imposent de citer la source et sa licence, ce que fait ce tableau et la fiche du modèle publié.
- **Modèle publié** : [`0biyohan/qwen3-1.7b-medical-triage-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-triage-fr) (public, Apache 2.0),
  modèle final (SFT v4 + DPO v5, ci-dessous) à la révision `4af91a7`, servi en local et dans le cloud. Version
  précédente (données v3) : révision `476fed3`.

## Entraînement (phase 2)

`notebook/sft_lora.ipynb` : SFT LoRA de `Qwen/Qwen3-1.7B-Base` (données v4), puis DPO LoRA sur le modèle SFT fusionné
(données v5, `DPO_DATA_VERSION`). Chaque run est suivi dans MLflow (DagsHub) et sauvegarde ses checkpoints et son
`run_config.json` dans `outputs/<nom du run>/`.

| Run | Réglages | Résultats (test) |
|---|---|---|
| `sft-lora-r16-v4-seed42` | rang 16, lr 2e-4, 1 époque, 1 536 tokens, perte sur toutes les réponses | QCM `mcqu` : 0,43 de bonnes réponses (réponse naïve : 0,21) |
| `dpo-lora-r16-v5-sigmoid+sft-beta0.1-lr5e-05-seed42` (retenu) | perte DPO + vraisemblance de la `chosen`, beta 0,1, lr 5e-5, 1 époque | préférences justes : 0,68 (fr), 0,74 (en) ; format 0,99 ; aucune régression (McNemar p = 0,36) ; propositions en trop 0,62 → 0,02 |

Modèle final fusionné : `outputs/dpo-lora-r16-v5-sigmoid+sft-beta0.1-lr5e-05-seed42/merged`, publié sur le Hub avec
les deux adaptateurs LoRA (`adapters/sft`, `adapters/dpo`) et la recette pour les réappliquer (fiche du modèle).
Servi par vLLM (`scripts/evaluate.py`, 380 QCM du test et 28 contrôles de sécurité) : 0,43 en `mcqu`, 4
recommandations dangereuses sur 12 cas pièges. Deux essais DPO
antérieurs (données v4, avec et sans terme de vraisemblance) sont comparés dans le rapport (partie 4.3).
Relancer seulement le DPO : section 1, la cellule des fonctions de génération (section 6), puis toute la section 7.

## Schéma des métadonnées

Chaque exemple des fichiers `data/v4/anonymized/*.anonymized.jsonl` (SFT) et `data/v5/anonymized/*.anonymized.jsonl`
(DPO) garde ses métadonnées, en plus du texte
d'entraînement (`messages` pour le SFT, `prompt` / `chosen` / `rejected` pour le DPO).

| Champ | Description | Jeux |
|---|---|---|
| `ids` | identifiants des questions d'origine de la ligne (plusieurs pour un cas MediQAl) | SFT |
| `id`, `prompt_id` | identifiant de la question d'origine | DPO MediQAl, DPO UltraMedical |
| `source` | jeu et configuration d'origine (`MediQAl-oeq`, `MediQAl-mcqu`, `MedQA`, `ChatDoctor`…) | tous |
| `lang` | langue : `fr` ou `en` | tous |
| `case_id` | identifiant du cas clinique ou de la question (empreinte SHA-1 du texte normalisé, 12 caractères) ; détermine le split (et l'usage SFT / DPO pour UltraMedical) | tous |
| `n_questions` | nombre de questions (réponses apprises) dans la ligne | SFT |
| `medical_subject` | spécialité médicale | MediQAl |
| `question_types` | type de chaque question (compréhension, raisonnement…) | SFT MediQAl |
| `golden_answers`, `golden_answer` | corrigé de chaque question (lettres des QCM) | SFT (QCM), DPO |
| `label_type` | type de paire d'origine (`hard`, `length`, `easy`, `human`) | DPO UltraMedical |
| `rejected_type` | erreur de la `rejected` : `omission`, `ajout` ou `substitution` | DPO MediQAl |
| `chosen_model`, `rejected_model` | LLM auteur de chaque réponse | UltraMedical |
| `chosen_score`, `rejected_score` | notes GPT-4 des réponses (de 1 à 5) | UltraMedical |
| `pair_check` | contrôle qui valide l'exemple : `golden_answer` (corrigé), `gpt4_score` (note GPT-4) ou `expert_review` (test revu par des experts) | UltraMedical |
| `constantes`, `urgence`, `triage` | critères de la sélection : constantes vitales dans le cas ou la question, vocabulaire d'urgence, paire proche du triage | DPO |
| `persons_replaced` | nombre de noms remplacés par `<PERSON>` lors de l'anonymisation | tous |

- **Niveau de confiance** : `pair_check`, `chosen_score` et `rejected_score` indiquent comment chaque réponse ou
  préférence a été validée.
- **Symptômes, antécédents, constantes vitales** : les sources ne les fournissent pas de façon structurée (ils
  figurent dans le texte libre des cas cliniques) ; la sélection ne fait que repérer leur présence. Les extraire en
  champs dédiés est un prérequis pour un futur jeu de triage.
- **Rapport d'anonymisation** : chaque fichier a son `*.report.json` (modèle spaCy, seuil, champs traités, comptes,
  détections écartées par motif).

## Anonymisation RGPD

Les notebooks anonymisent leurs JSONL avec `projet_14.rgpd` (Presidio et spaCy : `fr_core_news_md`, `en_core_web_md`) :
les noms de personnes sont remplacés par `<PERSON>`, sans table de correspondance. Le même module est disponible en
ligne de commande, pour un fichier SFT (`messages`) ou DPO (`prompt` / `chosen` / `rejected`) :

```bash
uv run rgpd-anonymize data/v4/sft_mediqal_mcqu_train.jsonl --lang fr --output-dir data/v4/anonymized
```

| Sortie | Contenu | Versionner |
|---|---|---|
| `data/<version>/anonymized/<nom>.anonymized.jsonl` | Dataset anonymisé, avec le nombre de noms remplacés par enregistrement (`persons_replaced`) | Non (`data/` ignoré, régénérable) |
| `data/<version>/anonymized/<nom>.report.json` | Paramètres (modèle spaCy, seuil, sources, rôles et champs traités), comptes, détections écartées par motif | Non (régénéré avec les données) |

- **Périmètre par jeu** (paramètres de `run`) : MediQAl `oeq`, cas et questions, et dans les réponses les noms placés
  après une civilité (`title_only_roles`) ; QCM MediQAl, cas et question seulement, propositions et réponses intactes
  (`keep_after="Propositions :"`) ; UltraMedical, source ChatDoctor seulement (seuls vrais échanges patients-médecins). DPO MediQAl : comme les QCM du SFT.
- **Règles propres au domaine médical** : civilités conservées (le sexe est une information clinique), éponymes
  (« maladie de Basedow »), sigles, marques de médicaments, mots sans majuscule et liste d'exclusion
  (`EXCLUDED_WORDS`), chiffres, paramètre suivi d'une valeur (« PaO2 96 mmHg ») et mot que spaCy ne reconnaît pas comme
  nom propre (« Justifiez ») ne sont pas pris pour des noms ; un nom après une civilité est toujours remplacé. Seuil : 0,5.
- **Version complète** : `projet_14.rgpd_extended` ajoute un journal d'audit par entité remplacée, un rapport
  Markdown et un échantillon de contrôle manuel (`reports/rgpd/`, échantillon non versionné). Elle a servi à
  l'étude initiale (`reports/rgpd/sft_mediqal_mcqu.*`).

## Démonstration locale (phase 3)

Le modèle fusionné (SFT + DPO, section 7.5 de `notebook/sft_lora.ipynb`) est servi par **vLLM** et exposé par une
API **FastAPI**. Les deux services tournent dans Docker (`docker-compose.yml`).

Prérequis : Docker avec accès au GPU (NVIDIA Container Toolkit). Le modèle est téléchargé, sans jeton, depuis le
dépôt public [`0biyohan/qwen3-1.7b-medical-triage-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-triage-fr),
à une révision fixée dans `docker-compose.yml` (variables `MODEL` et `MODEL_REVISION` pour en changer).

```bash
cp .env.example .env          # puis définir API_KEY (HF_TOKEN facultatif)
docker compose up -d          # premier lancement : image vLLM (~10 Go) et modèle (3,4 Go), mis en cache ensuite
export API_KEY=$(grep '^API_KEY=' .env | cut -d= -f2-)   # la clé n'est lue que par docker compose : la charger
curl localhost:8080/health -H "X-API-Key: $API_KEY"
```

Requête :

```bash
curl -X POST localhost:8080/triage -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{
  "age": 58, "sexe": "M", "motif": "douleur thoracique",
  "symptomes": "douleur constrictive depuis 40 minutes, irradiant dans le bras gauche, avec des sueurs",
  "antecedents": "tabagisme",
  "constantes": {"fc": 105, "pas": 150, "pad": 90, "spo2": 96}
}'
```

| Élément | Rôle |
|---|---|
| `POST /triage` | patient décrit en langage naturel (âge, motif, symptômes, antécédents, constantes) → niveau de priorité (maximale, modérée, différée), signes de gravité (constantes hors seuils), justification (dont la raison d'une absence de réorientation), conduite à tenir, questions complémentaires. Le modèle évalue le niveau sur un choix fermé (calibration expérimentale de son a priori avec `TRIAGE_CALIBRATION=1`, désactivée par défaut : elle sous-trie des urgences vitales) ; des garde-fous sur les constantes (seuils de démonstration) ne peuvent que le relever ; une liste fermée de motifs bénins (rhume, mal de gorge, petite plaie…), chez un adulte aux constantes complètes et normales sans signe d'alerte, donne une urgence différée. Clé obligatoire (`X-API-Key`) |
| `GET /health` | état de l'API ; avec la clé (`X-API-Key`), aussi celui de vLLM (sans clé, l'endpoint cloud n'est jamais interrogé, donc jamais réveillé) |
| `scripts/ui.py` | interface web de test (Gradio) : formulaire patient, cas de démonstration, niveau et explication ; cible locale ou cloud au choix dans la page. `uv run --with gradio python scripts/ui.py`, puis http://127.0.0.1:7860 (écoute sur la machine seulement : la clé d'API reste côté serveur) |
| `logs/audit.jsonl` | une ligne par requête : identifiant, horodatage, modèle, niveau retenu et sa source, latence, empreinte SHA-256 de l'entrée (le texte du patient n'est pas conservé) |
| `tests/test_api.py` | tests de l'API sans GPU (vLLM simulé), lancés aussi par la CI : `uv run --with-requirements api/requirements.txt --with pytest pytest tests` |
| `tests/integration/` | robustesse et audit de traçabilité contre la pile démarrée : `INTEGRATION_API_URL=http://localhost:8080 API_KEY=<clé> uv run --with-requirements api/requirements.txt --with pytest pytest tests/integration` (arrêt / redémarrage de vLLM avec `ROBUSTESSE_ARRET_VLLM=1`) |
| `scripts/evaluate_triage.py` | évaluation externe du niveau de triage, modèle seul, avec et sans calibration, sur 60 cas d'un jeu public (`reports/evaluation/triage_externe.csv`) : `uv run --with-requirements api/requirements.txt python scripts/evaluate_triage.py --vllm-url http://<vLLM>:8000` |
| `scripts/latency.py` | latence et débit à 1, 10 et 50 requêtes simultanées (API de bout en bout sur les cas de `scripts/triage_cases.json`, puis vLLM seul : TTFT, TPOT), niveaux conformes et urgences vitales sous-triées ; résultats dans MLflow (expérience `qwen3-1.7b-serving`) : `API_KEY=<clé> uv run python scripts/latency.py [--mlflow dagshub]` |
| `.github/workflows/ci.yml` | CI/CD : à chaque push, tests puis construction de l'image de l'API ; sur `master`, publication sur GHCR (`ghcr.io/obiyohan/triage-api:<commit>`), déploiement sur la machine GPU (runner auto-hébergé, étiquette `gpu`, dossier `~/p14-deploy`) et tests d'intégration contre la pile déployée |

**Déploiement continu** : secret `API_KEY` (et `HF_TOKEN`, facultatif) dans GitHub (*Settings → Secrets and variables → Actions*),
runner auto-hébergé avec l'étiquette `gpu` sur la machine qui sert la démonstration. Un nouveau commit sur `master`
remplace le conteneur de l'API sans redémarrer vLLM. Retour à une version précédente : relancer (*Re-run all jobs*)
le workflow d'un commit antérieur dans l'onglet *Actions*.

Le port de vLLM n'est pas publié : seule l'API est accessible. Démonstration technique, ne remplace pas un avis médical.

## Démo cloud (Hugging Face)

| Brique | Où | Coût |
|---|---|---|
| API FastAPI | Space Docker privé [`0biyohan/triage-api`](https://huggingface.co/spaces/0biyohan/triage-api) (CPU) | gratuit |
| Inférence vLLM | Inference Endpoint Hugging Face (GPU T4, AWS Irlande), mis en pause hors démonstration | 0,50 $/h, seulement allumé |
| Journal d'audit | logs du conteneur du Space (lignes `AUDIT {…}`, sans texte patient) | gratuit |

- **Déploiement de l'API** : `scripts/deploy_space.py` envoie au Space le code de l'API et le module du prompt
  (`deploy/space/` : `Dockerfile`, `README.md`) ; le Space reconstruit son image. La CI le fait à chaque commit sur
  `master` (job `deploy-cloud`, secret GitHub `HF_TOKEN_SPACE` : jeton en écriture limité au Space).
- **Modèle de l'endpoint** : `scripts/sync_endpoint.py` aligne le dépôt et la révision de l'endpoint sur ceux de
  `docker-compose.yml` (une pause est conservée, rien n'est fait s'il est à jour). La CI le lance après le Space
  (secret GitHub `HF_TOKEN_ENDPOINT` : jeton limité à la gestion des Inference Endpoints ; sans lui, étape ignorée).
  Nouvelle version du modèle : changer `MODEL_REVISION` dans `docker-compose.yml`, puis pousser sur `master`.
- **Configuration du Space** (*Settings → Variables and secrets*) : secrets `API_KEY`, `VLLM_URL` (URL de l'endpoint),
  `VLLM_API_KEY` (jeton Hugging Face en lecture) ; variables `MODEL_NAME=qwen3-medical`, `VLLM_TIMEOUT=300`.
- **Allumer et éteindre** : `uv run python scripts/cloud.py start` (relance l'endpoint et le Space, attend qu'ils soient
  prêts), `stop` (met l'endpoint en pause : plus de facturation GPU), `status`. Nom de l'endpoint : `HF_ENDPOINT_NAME`.
- **Démonstration** : `API_URL=https://0biyohan-triage-api.hf.space ./scripts/demo.sh`, ou l'interface `scripts/ui.py`
  (cible « Cloud »). Le Space étant privé, le
  script ajoute le jeton Hugging Face de la connexion locale ; la clé de l'API reste exigée.
