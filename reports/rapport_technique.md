---
title: "Agent IA de triage médical : preuve de concept"
subtitle: "Rapport technique et recommandations stratégiques"
author: "Johan"
date: "Novembre 2026"
lang: fr-FR
documentclass: article
fontsize: 11pt
geometry: "a4paper, margin=2.2cm"
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
monofontoptions: "Scale=0.85"
toc: true
toc-depth: 2
colorlinks: true
---

# 1. Contexte et périmètre

## 1.1 Le besoin

Le service des urgences du CHSA est en surcharge constante, surtout aux heures de pointe. Le manque d'effectifs infirmiers et de triage allonge les temps d'attente et augmente le risque de ne pas identifier assez tôt un cas critique.

Le projet étudie la faisabilité d'un **agent IA de triage médical**, chargé à terme de :

- collecter les symptômes des patients par un questionnaire adaptatif ;
- évaluer un niveau de priorité (urgence maximale, modérée ou différée) selon les protocoles médicaux ;
- expliquer son évaluation et ses recommandations ;
- s'intégrer au système d'information hospitalier (SIH) ;
- tracer chaque interaction pour les audits médicaux.

## 1.2 La démarche en trois phases

| Phase | Objectif | Réalisé dans ce POC |
|---|---|---|
| 1. Validation conceptuelle | Corpus médical bilingue, anonymisé, prêt pour l'entraînement | 3 jeux SFT et 2 jeux DPO, en français et en anglais, anonymisés avec Presidio |
| 2. Optimisation ciblée | Adapter Qwen3-1.7B-Base par SFT LoRA, puis l'aligner par DPO | Modèle SFT + DPO, publié sur Hugging Face Hub |
| 3. Projection industrielle | Servir le modèle (vLLM, FastAPI, Docker), automatiser le déploiement | API servie par vLLM sur GPU, CI/CD GitHub Actions, tests de latence et de robustesse |

## 1.3 Ce que le POC couvre, et ce qu'il ne couvre pas

Aucun jeu de données de triage aux urgences (description d'un patient associée à un niveau de priorité validé) n'est disponible sous une licence ouverte. Le POC s'appuie donc sur une tâche intermédiaire, mesurable automatiquement : **répondre à des questions médicales**.

- En **français**, des QCM sur des cas cliniques (MediQAl), avec un corrigé officiel. Le modèle doit choisir la ou les bonnes propositions.
- En **anglais**, des questions biomédicales avec des réponses rédigées et raisonnées (UltraMedical).

Ce choix permet de valider toute la chaîne technique (données, entraînement, alignement, mise en service, traçabilité) et de mesurer les connaissances médicales du modèle. Il ne permet **pas** d'évaluer la capacité de triage : le modèle n'a jamais appris à attribuer un niveau de priorité. Constituer un jeu de triage est la première étape de la feuille de route (partie 8).

## 1.4 Environnement

Tout le projet a été réalisé sur un poste de travail : carte graphique NVIDIA RTX 3080 (10 Go), sous Linux (WSL 2). Cette contrainte a guidé plusieurs choix : LoRA plutôt qu'un fine-tuning complet, limite de longueur des exemples, modèle de 1,7 milliard de paramètres.

# 2. Architecture : l'enchaînement des briques

## 2.1 Vue d'ensemble

Le schéma suit le parcours des données, des sources publiques jusqu'à la réponse de l'API.

```
 Sources publiques (Hugging Face)
   MediQAl (fr, QCM)       UltraMedical-Preference (en)
        |                            |
        v                            v
 [1] Préparation des données (notebooks Python)
     nettoyage, filtres qualité, répartition par cas clinique,
     anonymisation Presidio
        |
        v
     JSONL anonymisés : SFT (messages), DPO (prompt/chosen/rejected)
        |
        v
 [2] Entraînement (PyTorch, Transformers, PEFT, TRL)
     SFT LoRA  ->  DPO LoRA  ->  fusion des poids
        |                  suivi des runs : MLflow (DagsHub)
        v
 [3] Hugging Face Hub : modèle fusionné, dépôt privé, révision fixée
        |
        v
 [4] Service (Docker Compose, GPU)
     vLLM (inférence)  <---  API FastAPI (clé d'API, audit)  <--- client
        ^
        |
 [5] CI/CD GitHub Actions
     tests -> image de l'API (GHCR) -> déploiement -> tests d'intégration
```

## 2.2 Les briques

| Brique | Outils | Rôle | Livrable |
|---|---|---|---|
| Préparation | datasets, pandas | Charger, nettoyer, filtrer, découper | 15 fichiers JSONL |
| Anonymisation | Presidio, spaCy | Remplacer les noms de personnes | Données anonymisées |
| Entraînement | PEFT, TRL | SFT puis DPO avec LoRA | Adaptateurs, checkpoints |
| Suivi | MLflow, DagsHub | Hyperparamètres, métriques, runs | Historique des runs |
| Distribution | Hugging Face Hub | Héberger les poids fusionnés | Dépôt privé versionné |
| Inférence | vLLM 0.32 | Générer les réponses sur GPU | Serveur interne |
| Exposition | FastAPI | Valider, authentifier, tracer | `POST /qcm`, `GET /health` |
| Conteneurs | Docker Compose | Assembler vLLM et l'API | `docker-compose.yml` |
| CI/CD | GitHub Actions, GHCR | Tester, publier, déployer | Pipeline automatisé |

## 2.3 Le fil conducteur : un seul format, de bout en bout

Le principal risque d'une chaîne comme celle-ci est un **décalage de format** : le modèle reçoit en service un texte légèrement différent de celui qu'il a appris, et ses performances chutent sans erreur visible. Deux choix l'évitent.

**Un module partagé pour le prompt.** Le module Python `projet_14.mediqal` construit le message utilisateur (cas clinique, question, propositions, consigne). Il est utilisé à toutes les étapes :

| Étape | Utilisation |
|---|---|
| Préparation des données | construire les prompts d'entraînement |
| Évaluation | relire les lettres des réponses générées |
| API | construire le prompt à partir de la requête JSON |
| Tests et mesures | rejouer les QCM du jeu de test contre l'API |

Une fonction inverse retrouve le cas, la question et les propositions à partir d'un prompt. L'aller-retour est exact sur les 2 161 QCM de test.

**Un seul gabarit de conversation.** Le texte vu par le modèle suit le format ChatML de Qwen. Une particularité de Qwen3-*Base* a imposé un choix : ses balises `<|im_start|>` et `<|im_end|>` n'ont jamais été entraînées, et LoRA ne modifie pas les vecteurs des tokens. Le modèle ne peut donc pas apprendre à écrire `<|im_end|>`, et ne s'arrêterait jamais. Les réponses se terminent par `<|endoftext|>`, appris au pré-entraînement. Ce gabarit est sauvegardé avec le modèle, publié sur le Hub et appliqué par vLLM. Nous avons vérifié que vLLM produit exactement les mêmes tokens que l'entraînement : une même question donne la même réponse dans le notebook d'évaluation et à travers l'API.

## 2.4 Traçabilité et reproductibilité

Chaque maillon garde la trace de sa version, de sorte qu'une réponse de l'API peut être reliée aux données et aux réglages qui l'ont produite.

| Maillon | Trace conservée |
|---|---|
| Données | version (`v1` à `v3`) dans le nom des runs, graine fixe (42) |
| Entraînement | run MLflow : hyperparamètres, courbes, métriques ; `run_config.json` |
| Modèle | dépôt Hugging Face, révision fixée dans `docker-compose.yml` |
| API | image Docker étiquetée par le commit Git |
| Requêtes | journal d'audit : identifiant, horodatage, modèle, latence, réponse |

Le journal d'audit ne contient **aucun texte patient** : seule une empreinte SHA-256 de l'entrée est conservée. Elle permet de retrouver une requête sans stocker de données de santé.

## 2.5 Le parcours d'une requête

1. Le client envoie un QCM en JSON à `POST /qcm`, avec sa clé dans l'en-tête `X-API-Key`.
2. FastAPI vérifie la clé, puis valide l'entrée (longueurs, propositions non vides, aucune balise de gabarit).
3. L'API construit le prompt avec `projet_14.mediqal` et appelle vLLM (`temperature = 0`).
4. vLLM génère la réponse sur le GPU, en regroupant les requêtes simultanées.
5. L'API relit les lettres, vérifie que le format est exploitable, écrit une ligne dans le journal d'audit et renvoie la réponse, avec un avertissement sur les limites d'usage.

## 2.6 Le déploiement automatisé

À chaque modification du code, le pipeline GitHub Actions enchaîne trois étapes.

| Étape | Où | Contenu |
|---|---|---|
| `tests` | serveur GitHub | 17 tests de l'API, vLLM simulé (pas de GPU) |
| `image` | serveur GitHub | construction de l'image ; sur `master`, publication sur GHCR |
| `deploy` | machine GPU | mise à jour de la pile, puis tests d'intégration |

Le déploiement remplace le conteneur de l'API en quelques secondes, sans redémarrer vLLM. Le retour à une version précédente consiste à relancer le pipeline d'un commit antérieur.

# 3. Constitution des datasets SFT et DPO

## 3.1 Sources et critères de choix

| Source | Langue | Contenu | Licence |
|---|---|---|---|
| MediQAl | fr | QCM médicaux avec corrigé, avec ou sans cas clinique | CC-BY-4.0 |
| UltraMedical-Preference | en | Questions biomédicales, réponses de LLM notées | MIT |

Trois critères ont guidé le choix :

- **Bilingue** : le français pour la cible (le CHSA), l'anglais pour la richesse des ressources médicales.
- **Vérifiable** : un corrigé ou une note permet de contrôler automatiquement chaque exemple. C'est ce qui rend possible le principe « qualité plutôt que quantité ».
- **Licence ouverte** : CC-BY-4.0 et MIT autorisent la réutilisation et la modification, à condition de citer la source.

**MediQAl** (configurations `mcqu` et `mcqm`) rassemble des QCM rédigés par des enseignants en médecine. Les mauvaises propositions (les *distracteurs*) sont plausibles par construction, ce qui est précieux pour le DPO.

**UltraMedical-Preference** réunit des questions d'examens (MedQA, MedMCQA, PubMedQA), d'échanges patients-médecins (ChatDoctor), de manuels et de Wikipédia. Chacune a plusieurs réponses générées par des LLM (GPT-4, Llama-3, Mixtral, Qwen…) et notées par GPT-4. Son jeu de test (777 paires) a été revu par des experts biomédicaux.

## 3.2 Principes de construction

**Qualité plutôt que quantité.** Chaque exemple est vérifié avant d'être retenu : par le corrigé pour les QCM, par la note GPT-4 sinon. Les jeux sont ensuite plafonnés, plutôt que gonflés par des exemples douteux.

**Répartition par cas clinique.** Dans MediQAl, un même cas clinique sert à plusieurs questions, et les découpages officiels partagent des cas : **954 cas** se trouvent à la fois dans le train et dans le test officiels. Un modèle évalué sur ces cas les aurait déjà lus à l'entraînement, et ses scores seraient surestimés. MediQAl ne fournit pas d'identifiant de cas ; nous en avons calculé un.

- L'**identifiant** est l'empreinte SHA-1 du texte du cas, après normalisation des espaces et de la casse.
- Un **premier tirage**, dérivé de l'identifiant, affecte le cas au SFT ou au DPO (50 % chacun).
- Un **second tirage** l'affecte au train (70 %), à la validation (10 %) ou au test (20 %).

Les tirages sont déterministes : tous les notebooks obtiennent la même répartition. Un contrôle final vérifie, sur tous les fichiers MediQAl, qu'aucun cas n'est partagé entre SFT et DPO ni entre deux splits. Résultat : **1 039 cas** pour le SFT et **767 cas** pour le DPO, sans recouvrement.

**Pas de recouvrement entre SFT et DPO.** Le même principe s'applique à UltraMedical : les 3 056 questions utilisées par le DPO sont exclues du SFT.

**Formats standard.** Tous les jeux sont au format JSONL, lisible par la bibliothèque Hugging Face `datasets` :

- SFT : une liste `messages` (système, utilisateur, assistant) ;
- DPO : `prompt`, `chosen` (réponse préférée) et `rejected` (réponse moins bonne).

## 3.3 Les jeux SFT

| Jeu | Langue | Train | Validation | Test | Réponse attendue |
|---|---|---|---|---|---|
| MediQAl `mcqu` | fr | 2 454 | 342 | 749 | Une proposition |
| MediQAl `mcqm` | fr | 2 400 | 380 | 931 | Liste de 2 à 5 |
| UltraMedical | en | 1 500 | 300 | 500 | Texte raisonné |
| **Total** | | **6 354** | **1 022** | **2 180** | |

Le train compte 6 354 exemples, pour un objectif d'environ 5 000 paires.

**MediQAl `mcqu`** (une seule bonne réponse). Les 17 017 questions des trois splits officiels sont réunies, puis filtrées :

| Étape | Questions |
|---|---|
| Splits officiels réunis | 17 017 |
| Avec cas clinique, après dédoublonnage | 6 919 |
| Cas réservés au SFT : train / validation / test | 2 454 / 342 / 749 |

Seules les questions rattachées à un cas clinique sont gardées : elles décrivent un patient, ce qui est le plus proche du triage. Les 2 454 questions d'entraînement portent sur 684 cas distincts.

**MediQAl `mcqm`** (plusieurs bonnes réponses). Ce jeu prépare le DPO français, qui compare des listes de propositions : le modèle doit avoir appris ce format en SFT, car le DPO ajuste des préférences mais n'enseigne pas un nouveau format. Il réunit deux parties :

| Partie | Train | Validation | Test |
|---|---|---|---|
| Sans cas clinique (échantillon) | 1 000 | 200 | 500 |
| Avec cas clinique (cas du SFT) | 1 400 | 180 | 431 |

Les questions avec cas évitent que le modèle associe « cas clinique » à « une seule réponse », le seul format vu avec un cas dans `mcqu`.

**UltraMedical** (anglais). La meilleure réponse de chaque question sert de réponse attendue, après plusieurs filtres :

| Étape | Exemples (train) |
|---|---|
| Paires d'origine | 109 353 |
| Sans les questions du DPO | 103 582 |
| Réponses de qualité (corrigé ou note GPT-4) | 68 883 |
| Une réponse par question | 52 695 |
| Réponse entière sous la limite de longueur | 50 989 |
| Échantillon retenu | 1 500 |

Le filtre de longueur a une raison précise : une réponse coupée perdrait son token de fin, et le modèle apprendrait à ne pas s'arrêter.

**Le format des exemples MediQAl.** Le message utilisateur reprend le cas, la question, les propositions et une consigne sur le nombre de bonnes réponses :

```
Un homme de 45 ans [...] Le stade pathologique (PS) de ce patient est :

Propositions :
A. PS IIE
B. PS IIB
C. PS IIIA
D. PS IIIB
E. PS IVB

Une seule proposition est exacte.
```

La réponse attendue donne la lettre et le texte de la bonne proposition (`D. PS IIIB`), ou une liste (`- A. ...`) pour les réponses multiples. La lettre vient en premier : l'exactitude se mesure automatiquement en relisant les lettres.

## 3.4 Les jeux DPO

Le DPO apprend au modèle à préférer une réponse à une autre. Chaque exemple est une paire : pour un même prompt, une réponse `chosen` et une réponse `rejected`.

| Jeu | Langue | Train | Validation | Test |
|---|---|---|---|---|
| MediQAl `mcqm` | fr | 1 357 | 220 | 481 |
| UltraMedical-Preference | en | 2 000 | 300 | 756 |

À l'entraînement, les paires de plus de 1 024 tokens sont écartées (contrainte de mémoire GPU) : il reste 1 994, 297 et 750 paires anglaises.

**Le jeu français : des erreurs construites et garanties fausses.** Les 4 066 questions `mcqm` avec cas clinique sont filtrées sur les cas réservés au DPO (521 cas pour le train). Pour chaque question :

- la `chosen` est la liste des bonnes propositions, d'après le corrigé ;
- la `rejected` est la même liste avec **une seule erreur**, tirée au hasard parmi trois types.

| Type d'erreur | Construction | Paires (train) |
|---|---|---|
| Omission | une bonne proposition retirée | 495 |
| Ajout | une mauvaise proposition ajoutée | 428 |
| Substitution | une bonne remplacée par une mauvaise | 434 |

Les mauvaises propositions viennent des distracteurs rédigés par les enseignants : l'erreur est plausible, et le corrigé garantit qu'elle est fausse. Les propositions sont toujours affichées dans l'ordre A à E, pour que la position ne trahisse pas l'erreur. La `chosen` est la plus longue des deux réponses dans environ la moitié des paires : la longueur ne permet pas de deviner la bonne réponse.

**Le jeu anglais : des préférences vérifiées et équilibrées.**

| Étape | Paires (train) |
|---|---|
| Paires d'origine | 109 353 |
| Types `hard` et `length`, `chosen` mieux notée | 80 306 |
| Préférence vérifiée (corrigé, ou écart de note net) | 10 362 |
| Nettoyage et dédoublonnage | 8 879 |
| Sans les questions de validation et de test | 8 817 |
| Sous la limite de longueur | 8 577 |
| Échantillon équilibré sur la longueur | 2 000 |

Pour les sources QCM, une paire n'est gardée que si le corrigé confirme que la `chosen` est juste et la `rejected` fausse. Pour les autres sources, sans corrigé, la `chosen` doit avoir une très bonne note et un écart net avec la `rejected`. Le champ `pair_check` garde la trace du contrôle appliqué.

**L'équilibrage sur la longueur** corrige un biais découvert au premier entraînement. Après filtrage, la `chosen` n'était la plus longue que dans un quart des paires. Le modèle avait surtout appris que « la réponse la plus courte est la meilleure » : 0,82 de préférences justes en validation, mais 0,51 en test, soit le hasard. Le train et la validation contiennent désormais autant de paires où la `chosen` est la plus longue que de paires où elle est la plus courte.

Le **test** est gardé entier et non équilibré : revu par des experts, il représente la distribution réelle (la `chosen` y est la plus longue dans 70 % des paires). L'évaluation le découpe selon la longueur, pour détecter un biais résiduel.

## 3.5 Anonymisation et conformité RGPD

**L'outil.** Presidio (Microsoft) détecte les noms de personnes avec les modèles de langue spaCy (`fr_core_news_md`, `en_core_web_md`) et les remplace par la balise `<PERSON>`. Seules les détections de score supérieur ou égal à 0,5 sont retenues. Des règles propres au domaine médical écartent ensuite les faux positifs les plus fréquents :

| Motif d'exclusion | Exemple | Détections écartées |
|---|---|---|
| Mot sans majuscule | « fièvre », « dyspnée » | 6 225 |
| Sigle | « HTA », « ECG » | 516 |
| Éponyme médical | « maladie de Basedow » | 513 |
| Marque déposée | nom commercial d'un médicament | 197 |
| Mot exclu (liste) | termes médicaux fréquents | 104 |
| **Total** | | **7 555** |

Sans ces règles, plus de 7 500 termes médicaux auraient été remplacés par `<PERSON>`.

**Un périmètre ciblé.** Anonymiser tout le texte aurait détruit l'information médicale : Presidio prend de nombreux termes médicaux pour des noms (« Angiographie », « Érythrose palmaire », gènes « Akt » ou « FOXO1 »). Seuls les champs qui peuvent décrire un patient réel sont donc traités :

| Jeu | Champs anonymisés | Raison |
|---|---|---|
| MediQAl | cas clinique et question | ils décrivent le patient |
| MediQAl | propositions et réponses : non | énoncés rédigés par les enseignants |
| UltraMedical | source ChatDoctor seulement | seuls vrais échanges patients-médecins |

**Les volumes traités** (train, validation et test réunis) :

| Jeu | Exemples | Modifiés | Noms remplacés |
|---|---|---|---|
| SFT MediQAl `mcqu` | 3 545 | 1 209 | 2 157 |
| SFT MediQAl `mcqm` | 3 711 | 722 | 1 249 |
| DPO MediQAl `mcqm` | 2 058 | 674 | 1 235 |
| SFT UltraMedical | 2 300 | 96 | 203 |
| DPO UltraMedical | 3 056 | 174 | 464 |

Pour UltraMedical, seuls les exemples ChatDoctor sont analysés : 190 en SFT et 290 en DPO. Les remplacements comptés dans MediQAl peuvent inclure des faux positifs résiduels : chaque notebook affiche le contexte de chaque balise `<PERSON>` pour une relecture.

**Contrôles.** Après anonymisation, des contrôles vérifient que les propositions et les réponses MediQAl sont intactes (aucune balise `<PERSON>` sur les 9 314 exemples), et que les paires DPO restent distinctes (`chosen` différente de `rejected`). Un rapport par fichier enregistre les paramètres (modèle spaCy, seuil, champs traités), les comptes et les détections écartées par motif.

**Justification RGPD.**

- **Minimisation** : seuls les champs susceptibles de contenir des données personnelles sont traités ; aucune donnée n'est ajoutée.
- **Pseudonymisation irréversible** : les noms sont remplacés par une balise, sans table de correspondance.
- **Traçabilité** : chaque fichier anonymisé a son rapport (paramètres, comptes, motifs d'exclusion), et chaque exemple garde le nombre de noms remplacés (`persons_replaced`).
- **Données dérivées non publiées** : les jeux anonymisés ne sont pas versionnés dans Git. Ils se régénèrent à partir des sources et des notebooks.
- **En service** : le journal d'audit de l'API ne conserve que l'empreinte de l'entrée, jamais le texte du cas.

## 3.6 Trois itérations sur les données

Les données ont évolué au fil des entraînements : chaque version corrige un problème révélé par la précédente.

| Version | Problème observé | Correction |
|---|---|---|
| v1 | Propositions absentes du prompt : aucune lettre exploitable | Propositions A à E dans le prompt, réponse « lettre + texte » |
| v1 | 954 cas partagés entre train et test officiels | Répartition par cas clinique |
| v1 | DPO anglais : préférence apprise pour la réponse courte | Équilibrage sur la longueur |
| v2 | 25 % de réponses à plusieurs lettres en `mcqu` | Consigne « une seule / plusieurs propositions » |
| v3 | — | Version entraînée et servie |

L'effet de la dernière correction est direct : en `mcqu`, la part de réponses à plusieurs lettres passe de 25 % à 0 %, et l'exactitude du modèle SFT de 0,32 à 0,39 (200 questions de test).

## 3.7 Schéma des métadonnées

Chaque exemple garde ses métadonnées, en plus du texte d'entraînement.

| Champ | Description | Jeux |
|---|---|---|
| `id` | identifiant de la question d'origine | MediQAl |
| `prompt_id` | identifiant de la question d'origine | UltraMedical |
| `source` | jeu et configuration d'origine | tous |
| `lang` | langue (`fr`, `en`) | tous |
| `case_id` | identifiant du cas clinique | MediQAl |
| `medical_subject` | spécialité médicale | MediQAl |
| `question_type` | type de question | MediQAl |
| `golden_answer` | corrigé (QCM) ou réponse de référence | tous |
| `rejected_type` | type d'erreur de la `rejected` | DPO fr |
| `label_type` | type de paire d'origine | DPO en |
| `chosen_model`, `rejected_model` | LLM auteur de chaque réponse | UltraMedical |
| `chosen_score`, `rejected_score` | notes GPT-4 des réponses | UltraMedical |
| `pair_check` | contrôle appliqué à l'exemple | DPO, SFT en |
| `persons_replaced` | noms remplacés par l'anonymisation | tous |

Les champs `pair_check`, `chosen_score` et `rejected_score` jouent le rôle d'un **niveau de confiance** : ils disent comment chaque préférence a été validée.

Le schéma demandé pour la phase 1 prévoyait aussi des champs **symptômes, antécédents et constantes vitales**. Les sources utilisées ne les fournissent pas de façon structurée : ces informations figurent dans le texte libre des cas cliniques. Les structurer est un prérequis pour un jeu de triage, et fait partie de la feuille de route.
