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

# Résumé

**Objectif.** Le CHSA souhaite évaluer la faisabilité d'un agent IA d'aide au triage pour son service des urgences. Ce POC construit toute la chaîne technique nécessaire, des données jusqu'à un service déployé, avec un modèle compact : Qwen3-1.7B-Base.

**Ce qui a été construit.**

- **Données** : un corpus médical bilingue, construit à partir de deux sources ouvertes (MediQAl en français, UltraMedical en anglais) et **sélectionné pour le triage** : chaque exemple décrit un patient, en priorité avec des constantes et des signes d'urgence. Il compte 4 998 paires SFT (2 998 en français, 2 000 en anglais) et 3 712 paires DPO vérifiées (2 156 en français, 1 556 en anglais, dont 756 paires de test revues par des experts). Il est réparti par cas clinique pour éviter toute fuite entre entraînement et test, et anonymisé avec Presidio.
- **Modèle** : un fine-tuning supervisé (SFT) avec LoRA, puis un alignement par préférences (DPO), sur une seule carte graphique de 10 Go. Le DPO retenu est le troisième essai, qui ajoute à la perte DPO un terme de vraisemblance de la bonne réponse. Chaque run est tracé dans MLflow. Le modèle final est publié sur Hugging Face Hub et servi par la démonstration.
- **Service** : une API de triage (patient décrit en langage naturel → niveau de priorité expliqué, avec des questions complémentaires), protégée par clé, servie par vLLM, avec un journal d'audit sans donnée patient. Elle est déployée en local (Docker, RTX 3080) et dans le cloud (Space et Inference Endpoint Hugging Face). Un pipeline GitHub Actions teste, publie et redéploie automatiquement chaque nouvelle version.

**Résultats clés.**

| Indicateur | Résultat |
|---|---|
| QCM à réponse unique (389 questions) | 0,44 de bonnes réponses, contre 0,21 pour une réponse naïve |
| DPO : préférences justes (test) | 0,68 en français, 0,74 en anglais (0,50 = hasard) |
| DPO : effet sur les réponses | aucune régression ; propositions cochées en trop : 0,62 → 0,02 par question |
| Contrôles de sécurité (12 cas) | 4 recommandations dangereuses (3 pour le modèle précédent) ; entités inventées jamais reconnues |
| Latence (1 requête, médiane) | 82 ms en local ; 0,55 s dans le cloud, réseau compris |
| Débit (50 requêtes simultanées) | 64 requêtes par seconde (local) |

**Conclusion.** La faisabilité technique est démontrée : la chaîne complète fonctionne, est reproductible et traçable, et le service répond en moins d'une seconde. La valeur clinique, en revanche, n'est pas démontrée. Le modèle n'a pas appris le triage, faute de données adaptées, et il commet encore des erreurs dangereuses. Il ne doit pas être utilisé auprès de patients.

**Recommandation.** Avant de passer à un modèle plus grand, constituer avec les équipes du CHSA un jeu de cas de triage annotés, puis valider le système en conditions réelles, sans effet sur la décision des soignants (partie 8).

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
| 1. Validation conceptuelle | Corpus médical bilingue, anonymisé, prêt pour l'entraînement | 4 jeux SFT et 2 jeux DPO (fr, en), sélectionnés pour le triage, anonymisés avec Presidio |
| 2. Optimisation ciblée | Adapter Qwen3-1.7B-Base par SFT LoRA, puis l'aligner par DPO | Modèle SFT + DPO, trois essais de DPO comparés, publié sur Hugging Face Hub |
| 3. Projection industrielle | Servir le modèle (vLLM, FastAPI, Docker), automatiser le déploiement | API servie par vLLM sur GPU, CI/CD GitHub Actions, tests de latence et de robustesse |

## 1.3 Périmètre du POC

Le POC entraîne le modèle à **répondre à des questions médicales sur un patient décrit**.

- En **français**, des cas cliniques rédigés par des enseignants (MediQAl) : questions ouvertes à réponse rédigée, et QCM avec corrigé officiel.
- En **anglais**, des vignettes d'examen et des questions de patients, avec des réponses raisonnées (UltraMedical).

Les cas sont sélectionnés pour leur proximité avec le triage (constantes, signes d'urgence). Ce choix permet de valider toute la chaîne technique (données, entraînement, alignement, mise en service, traçabilité) et de mesurer les connaissances médicales du modèle. Il ne permet **pas** d'évaluer la capacité de triage : aucune source n'attribue de niveau de priorité. Constituer un jeu de triage est la première étape de la feuille de route (partie 8).

## 1.4 Environnement

Tout le projet a été réalisé sur un poste de travail : carte graphique NVIDIA RTX 3080 (10 Go), sous Linux (WSL 2).

# 2. Architecture : l'enchaînement des briques

## 2.1 Vue d'ensemble

Le schéma suit le parcours des données, des sources publiques jusqu'à la réponse de l'API.

```
 Sources publiques (Hugging Face)
   MediQAl (fr, cas cliniques)   UltraMedical-Preference (en)
        |                            |
        v                            v
 [1] Préparation des données (notebooks Python)
     nettoyage, filtres qualité, sélection pour le triage,
     répartition par cas clinique, anonymisation Presidio
        |
        v
     JSONL anonymisés : SFT (messages), DPO (prompt/chosen/rejected)
        |
        v
 [2] Entraînement (PyTorch, Transformers, PEFT, TRL)
     SFT LoRA  ->  DPO LoRA  ->  fusion des poids
        |                  suivi des runs : MLflow (DagsHub)
        v
 [3] Hugging Face Hub : modèle fusionné, dépôt public, révision fixée
        |
        v
 [4] Service : local (Docker Compose, RTX 3080)
               ou cloud (Inference Endpoint T4 + Space Hugging Face)
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
| Distribution | Hugging Face Hub | Héberger les poids fusionnés | Dépôt public versionné |
| Inférence | vLLM 0.32 | Générer les réponses sur GPU | Serveur interne |
| Exposition | FastAPI | Valider, authentifier, décider, tracer | `POST /triage`, `GET /health` |
| Conteneurs | Docker Compose | Assembler vLLM et l'API | `docker-compose.yml` |
| Démo cloud | HF Inference Endpoints, Spaces | Servir le modèle et l'API en ligne | Endpoint T4, Space privé |
| CI/CD | GitHub Actions, GHCR | Tester, publier, déployer | Pipeline automatisé |

## 2.3 Le déploiement automatisé

À chaque modification du code, le pipeline GitHub Actions enchaîne quatre étapes.

| Étape | Où | Contenu |
|---|---|---|
| `tests` | serveur GitHub | 42 tests de l'API et des métadonnées du Space, vLLM simulé (pas de GPU) |
| `image` | serveur GitHub | construction de l'image ; sur `master`, publication sur GHCR |
| `deploy` | machine GPU | mise à jour de la pile locale, puis tests d'intégration |
| `deploy-cloud` | serveur GitHub | mise à jour de l'API sur le Space, puis test de fumée |

Le déploiement remplace le conteneur de l'API en quelques secondes, sans redémarrer vLLM. Le retour à une version précédente consiste à relancer le pipeline d'un commit antérieur.

# 3. Constitution des datasets SFT et DPO

## 3.1 Sources et critères de choix

| Source | Langue | Contenu | Licence | Usage |
|---|---|---|---|---|
| MediQAl | fr | Cas cliniques rédigés par des enseignants en médecine, avec questions ouvertes (`oeq`) et QCM (`mcqu`, `mcqm`) corrigés | CC-BY-4.0 | SFT et DPO |
| UltraMedical-Preference | en | Vignettes d'examen (MedQA, MedMCQA), questions de patients (ChatDoctor) ; réponses de LLM notées par GPT-4, test revu par des experts | MIT | SFT et DPO |

Trois critères ont guidé le choix :

- **Proche du triage** : un patient est décrit (âge, symptômes, constantes) et une décision est attendue.
- **Bilingue** : le français pour la cible (le CHSA), l'anglais pour la richesse des ressources médicales.
- **Vérifiable** : un corrigé, une réponse d'enseignant ou une note permet de contrôler chaque exemple.

Deux autres jeux ont été étudiés et écartés. **MedQuAD** ne contient que des questions générales sur les maladies (« What are the symptoms of X? ») : aucune de ses 5 948 questions reprises dans UltraMedical ne décrit un patient, et il n'a qu'une réponse par question, sans réponse rejetée pour le DPO. **FrenchMedMCQA** rassemble des QCM de pharmacie sans cas clinique.

## 3.2 Principes de construction

**Qualité plutôt que quantité.** Chaque exemple est vérifié avant d'être retenu, puis les jeux sont plafonnés : environ 5 000 paires pour le SFT, un plafond (et non un objectif) pour le DPO.

**Une ligne par cas clinique.** Un cas MediQAl porte plusieurs questions qui forment un récit : examens, résultats, évolution, traitement. Chaque cas devient une conversation : le cas et la première question, la réponse, la question suivante, sa réponse, et ainsi de suite. À l'entraînement, la perte porte sur **toutes les réponses** de l'assistant, chacune avec l'historique qui la précède : c'est l'équivalent d'un exemple par question, sans dupliquer le texte (quatre fois moins de tokens). Un cas trop long (plus de 3 500 caractères) est coupé en tranches, chacune reprenant le cas en tête.

**Répartition par cas clinique.** Les découpages officiels de MediQAl partagent des cas entre train et test : un modèle évalué sur ces cas les aurait déjà lus. L'identifiant d'un cas est l'empreinte SHA-1 de son texte normalisé ; un tirage déterministe qui en dérive l'affecte au train (70 %), à la validation (10 %) ou au test (20 %), de façon identique dans tous les notebooks. Le DPO n'utilise que des cas MediQAl non retenus pour le SFT ; pour UltraMedical, un premier tirage attribue chaque question au SFT ou au DPO (moitié chacun). Aucun cas ni aucune question n'est commun aux deux.

**Formats standard.** JSONL lisible par Hugging Face `datasets` : `messages` pour le SFT, `prompt` / `chosen` / `rejected` pour le DPO. Chaque exemple garde ses métadonnées, documentées dans le README. Le jeu est public sur Hugging Face Hub ([`0biyohan/triage-medical-bilingual`](https://huggingface.co/datasets/0biyohan/triage-medical-bilingual), version `v1.0`, celle du modèle final), avec sources, licences et anonymisation.

## 3.3 Sélection pour le triage

Le corpus complet compte près de 30 000 questions en français. On garde les **cas les plus proches du triage**, par cas entier, avec les mêmes critères dans les deux langues :

1. **Patient décrit** (obligatoire) : un cas clinique, une vignette (« A 45-year-old man… ») ou un patient qui décrit ses symptômes.
2. **Score de pertinence** (0 à 2) : présence de **constantes** (PA, FC, SpO₂, température, Glasgow…) et de vocabulaire d'**urgence** ou de conduite à tenir (hospitalisation, gravité, choc, détresse…).
3. **Budget** de questions, réparti entre train, validation et test : cas de score 2, puis 1, dans un ordre pseudo-aléatoire fixe.
4. **Plafond par spécialité** (15 % du budget) ou par source (40 %), pour garder la diversité.

| Jeu | Questions disponibles | Patient, score ≥ 1 | Retenues (cas) | Train / val. / test |
|---|---|---|---|---|
| MediQAl `oeq` | 3 632 | 2 672 | 1 098 (204) | 770 / 109 / 219 |
| MediQAl `mcqu` | 16 212 | 4 430 | 1 200 (293) | 840 / 120 / 240 |
| MediQAl `mcqm` | 10 006 | 2 342 | 700 (251) | 490 / 70 / 140 |
| UltraMedical | 12 038 | 3 518 | 2 000 | 1 400 / 200 / 400 |
| **Total** | | | **4 998** | **3 500 / 499 / 999** |

Plus de 90 % des cas retenus contiennent du vocabulaire d'urgence ; les constantes figurent dans 67 à 100 % des cas français selon le split, et dans 56 % des questions anglaises. Le plafond limite la Pharmacie, plus de 40 % des questions ouvertes disponibles, à 161 questions ; Réanimation, Pneumologie, Cardiologie et Médecine d’urgence suivent.

## 3.4 Les jeux SFT

**MediQAl `oeq`** (questions ouvertes). Publiées dans une seule partie (4 969 questions), redécoupées par cas. Les questions qui renvoient à une image (122) et la Cardiologie pédiatrique (1 206 questions, loin des urgences adultes) sont écartées. Les réponses rédigées par les enseignants apprennent au modèle à **expliquer**, ce que les QCM ne font pas.

**MediQAl `mcqu` et `mcqm`** (QCM à une ou plusieurs bonnes réponses). Les trois parties officielles sont réunies (17 017 questions en `mcqu`), puis redécoupées par cas. Chaque question porte ses propositions A à E et une consigne (« Une seule proposition est exacte », « Plusieurs propositions sont exactes »). La mention « (cochez la réponse fausse) » est conservée : elle inverse le sens de la question. La réponse attendue donne la lettre et le texte (`C. …`), ou une liste : la lettre vient en premier, ce qui rend l'exactitude mesurable automatiquement.

**UltraMedical** (anglais). La meilleure réponse de chaque question sert de réponse attendue. Des 111 541 paires d'origine (hors questions du test expert), on garde 55 964 questions dont la réponse est vérifiée (corrigé du QCM) et notée au moins 4,5 par GPT-4, puis 24 269 issues de sources cliniques (MedQA, MedQA-Evol, MedMCQA, ChatDoctor), 12 038 dans la moitié SFT, et 2 000 après la sélection pour le triage.

Le filtre de longueur a une raison précise : une réponse coupée perdrait son token de fin, et le modèle apprendrait à ne pas s'arrêter. Avec une longueur maximale d'entraînement de 1 536 tokens, aucune ligne n'est coupée.

## 3.5 Les jeux DPO

Le DPO apprend au modèle à préférer une réponse à une autre : pour un même prompt, une réponse `chosen` et une réponse `rejected`. Les deux jeux suivent les mêmes principes que le SFT : orientés triage, vérifiés, sans cas ni question commune avec le SFT.

| Jeu | Langue | Train | Validation | Test | Vérification |
|---|---|---|---|---|---|
| MediQAl `mcqu` et `mcqm` | fr | 1 507 | 201 | 448 | corrigé |
| UltraMedical-Preference | en | 727 | 73 | 756 | corrigé ou note ; test revu par des experts |

Les jeux DPO en sont à leur **version 5** ; le SFT reste en version 4. Chaque version corrige un raccourci appris par l'essai précédent (partie 4.3).

**Le jeu français : des erreurs construites et garanties fausses.** Les QCM des cas cliniques non retenus pour le SFT sont sélectionnés avec les mêmes critères de triage (750 questions `mcqu` et 1 500 `mcqm` au plus). Le prompt reprend le format du SFT : le cas, les questions précédentes avec leurs réponses de référence, puis la question. La `chosen` est le corrigé ; la `rejected` contient **une seule erreur**, prise parmi les distracteurs rédigés par les enseignants : plausible, et garantie fausse par le corrigé.

En `mcqu`, la `rejected` est un distracteur à la place de la bonne réponse (726 paires). En `mcqm` (1 430 paires), l'erreur est une **substitution** (une bonne proposition remplacée par une mauvaise, 716 paires), une **omission** (une bonne proposition retirée, 356) ou un **ajout** (une mauvaise ajoutée, 358).

- **Consigne respectée** : une omission n'est construite que si la `rejected` contient encore au moins deux propositions. En v3, 81 paires de test contredisaient la consigne, ce qui rendait l'erreur trop facile à repérer.
- **Le nombre de propositions ne trahit pas la bonne réponse** : moitié de substitutions (autant de propositions de part et d'autre), autant d'omissions que d'ajouts. En v3 et v4, avec un tiers de chaque type, le DPO avait appris à « cocher moins ».

**Le jeu anglais : des préférences vérifiées, courtes et équilibrées.** Des 111 585 paires d'origine, 24 772 relèvent de la moitié DPO et des sources du SFT, hors test expert. 4 195 ont une préférence vérifiée et 2 431 décrivent un patient. 800 forment l'échantillon final, avec une paire par question.

- **Vérification** : pour les QCM, la `chosen` donne la lettre du corrigé et la `rejected` une autre (790 paires) ; pour ChatDoctor, note d'au moins 4,5 et écart d'au moins 2 points (10 paires).
- **Réponses de 2 000 caractères au plus** (675 paires écartées, dont 381 ChatDoctor) : en v4, les réponses anglaises, environ 30 fois plus longues qu'une paire française, dominaient l'apprentissage, et leur style débordait sur le français.
- **Équilibrage sur la longueur** : en v3, le modèle avait appris que « la réponse la plus courte est la meilleure » (0,82 de préférences justes en validation, 0,51 en test). Le jeu compte désormais 400 paires dans chaque sens.
- **Test** : le jeu revu par des experts (756 paires), gardé entier, dont 160 paires proches du triage mesurées à part.

## 3.6 Anonymisation et conformité RGPD

**L'outil.** Presidio (Microsoft) détecte les noms de personnes avec spaCy (`fr_core_news_md`, `en_core_web_md`) et une règle sur les civilités, puis les remplace par `<PERSON>` (score minimal 0,5). Des règles médicales écartent les faux positifs : mots sans majuscule, sigles, éponymes (« maladie de Basedow »), marques, chiffres, paramètre suivi d'une valeur (« PaO2 96 mmHg »), mot qui n'est pas un nom propre pour spaCy (« Justifiez »). Ces deux dernières règles ont été ajoutées en v4 après un contrôle par sondage des balises `<PERSON>` : sur les questions ouvertes, environ la moitié des remplacements étaient des faux positifs, jusque dans les réponses attendues (« <PERSON> coli » pour *Escherichia coli*). Un nom placé après une civilité est toujours remplacé, même en majuscules (« Monsieur DURAND »).

**Un périmètre ciblé.** Anonymiser tout le texte détruirait l'information médicale. Seuls les champs qui peuvent décrire un patient réel sont traités :

| Jeu | Champs anonymisés | Raison |
|---|---|---|
| MediQAl `oeq` | cas et questions ; dans les réponses, noms après une civilité seulement | les réponses reprennent parfois le nom du patient, mais leurs autres détections sont des termes médicaux |
| MediQAl QCM (SFT et DPO) | cas et question ; propositions et réponses non | énoncés rédigés par les enseignants |
| UltraMedical | source ChatDoctor seulement | seuls vrais échanges patients-médecins |

**Les volumes traités** (train, validation et test réunis) :

| Jeu | Lignes | Analysées | Modifiées | Noms remplacés |
|---|---|---|---|---|
| SFT MediQAl `oeq` | 369 | 369 | 211 | 564 |
| SFT MediQAl `mcqu` | 353 | 353 | 123 | 225 |
| SFT MediQAl `mcqm` | 263 | 263 | 85 | 164 |
| SFT UltraMedical | 2 000 | 467 (ChatDoctor) | 200 | 406 |
| DPO MediQAl | 2 156 | 2 156 | 579 | 1 053 |
| DPO UltraMedical | 1 556 | 102 (ChatDoctor) | 57 | 136 |

Au total, 4 759 détections ont été écartées par les règles, dont 3 074 mots sans majuscule, 568 mots qui ne sont pas des noms propres, 319 éponymes et 282 sigles. Un cas qui revient dans plusieurs paires DPO compte ses noms à chaque fois.

**Contrôles.** Après anonymisation, les contrôles finaux vérifient qu'aucun cas n'est partagé entre splits, que chaque ligne MediQAl part d'un cas clinique, que les propositions et réponses des QCM sont intactes, que les paires DPO restent distinctes et qu'aucune question n'est commune au SFT et au DPO. Un rapport par fichier enregistre les paramètres, les comptes et les détections écartées par motif.

**Justification RGPD.**

- **Minimisation** : seuls les champs susceptibles de contenir des données personnelles sont traités ; aucune donnée n'est ajoutée.
- **Pseudonymisation irréversible** : les noms sont remplacés par une balise, sans table de correspondance.
- **Traçabilité** : chaque fichier anonymisé a son rapport, et chaque exemple garde le nombre de noms remplacés (`persons_replaced`).
- **Données dérivées non publiées** : les jeux anonymisés ne sont pas versionnés dans Git. Ils se régénèrent à partir des sources et des notebooks.
- **En service** : le journal d'audit de l'API ne conserve que l'empreinte de l'entrée, jamais le texte du cas.

# 4. Entraînement

Le modèle final est entraîné sur les données décrites en partie 3 : SFT sur la version 4, DPO sur la version 5. Tout se trouve dans `notebook/sft_lora.ipynb`. Le modèle précédent, entraîné sur la version 3 (une question par exemple, sans questions ouvertes), sert de point de comparaison pour le modèle servi (parties 5.4 et 6).

## 4.1 Méthode : SFT puis DPO, avec LoRA

L'entraînement se fait en deux temps, sur le même modèle de base, `Qwen/Qwen3-1.7B-Base`.

- **SFT** (*Supervised Fine-Tuning*) : le modèle lit des questions et les réponses attendues, et apprend à les reproduire. La perte ne porte que sur la réponse, pas sur la question.
- **DPO** (*Direct Preference Optimization*) : le modèle lit des paires de réponses à une même question, et apprend à rendre la réponse préférée plus probable que l'autre. Le modèle SFT sert de **référence** : le DPO mesure ce que le modèle préfère *de plus* que le SFT, ce qui l'empêche de s'en éloigner trop.

Les deux étapes utilisent **LoRA** : les poids du modèle sont figés, et seules de petites matrices ajoutées à chaque couche sont entraînées. Avec un rang de 16 sur toutes les couches linéaires, cela représente 17,4 millions de paramètres, soit 1 % du modèle. Un fine-tuning complet demanderait environ 27 Go de mémoire GPU ; avec LoRA, l'entraînement tient sur les 10 Go de la RTX 3080.

Pour le DPO, l'adaptateur SFT est d'abord fusionné dans les poids, puis un nouvel adaptateur est entraîné par-dessus. La référence s'obtient en désactivant ce nouvel adaptateur : il n'est pas nécessaire de charger un second modèle en mémoire.

## 4.2 Réglages

| Réglage | SFT | DPO |
|---|---|---|
| Données (train) | 2 086 lignes, 3 500 questions (v4) | 2 234 paires (v5) |
| LoRA | rang 16, alpha 32, dropout 0,05 | identique |
| Époques | 1 | 1 |
| Learning rate | 2e-4, décroissance en cosinus | 5e-5, décroissance en cosinus |
| Montée progressive | 3 % des pas | 10 % des pas |
| Batch effectif | 16 lignes (2 × 8) | 16 paires (2 × 8) |
| Longueur maximale | 1 536 tokens | 1 024 tokens |
| Perte | toutes les réponses de l'assistant | DPO (beta = 0,1) + vraisemblance de la `chosen`, poids 1 et 1 |
| Pas d'entraînement | 131 | 140 |
| Durée (RTX 3080) | 19 min | 31 min |

Trois réglages ont été ajustés au fil des runs :

- **Une seule époque de SFT** : avec deux époques (v3), la perte de validation atteignait son minimum à la fin de la première, puis remontait légèrement. La seconde époque coûtait 30 minutes sans rien apporter.
- **Learning rate du DPO à 5e-5** : la valeur usuelle en DPO complet (5e-6) ne produisait presque aucun apprentissage avec LoRA, qui n'entraîne qu'1 % des poids. La marge entre réponses préférées et rejetées restait proche de zéro.
- **Terme de vraisemblance dans la perte DPO** (essais 2 et 3 ci-dessous) : le DPO seul compare deux réponses, sans rien imposer sur leur forme. Ajouter la perte du SFT sur la `chosen` maintient le modèle sur le format appris.

## 4.3 Déroulement

**SFT.** La perte de validation passe de 1,090 (modèle de base) à 0,972, et la part de tokens bien prédits de 0,71 à 0,73. Elle baisse régulièrement et se stabilise en fin d'époque, sans remonter ; le meilleur checkpoint est le dernier. Sur le test, elle est identique à la validation (0,973) : le modèle généralise, sans apprendre par cœur.

Les pertes par source (0,09 à 0,13 sur les QCM français, 0,85 sur l'anglais, 2,14 sur les questions ouvertes) ne suffisent pas à juger le modèle : dans un QCM, le texte des propositions est dans le prompt, et la seule vraie difficulté est la lettre. D'où l'évaluation sur les réponses générées (partie 5).

**DPO : trois essais sur le même modèle SFT.** Chaque essai est mesuré sur les paires de test et sur les réponses générées aux QCM français.

| | Essai 1 | Essai 2 | **Essai 3 (retenu)** |
|---|---|---|---|
| Données DPO | v4 | v4 | v5 |
| Perte | DPO | DPO + SFT | DPO + SFT |
| Préférences justes, français | 0,63 | 0,62 | **0,68** |
| – omission | 0,41 | 0,38 | 0,53 |
| – ajout | 0,87 | 0,87 | 0,71 |
| – substitution | 0,64 | 0,64 | 0,70 |
| Préférences justes, anglais (750 paires) | 0,78 | 0,78 | 0,74 |
| Réponses au format attendu | 0,87 | 0,98 | **0,99** |
| Réponses de plus de 300 caractères | 10 % | 0 | 0 |

- **Essai 1** : des préférences apprises, mais des réponses dégradées : une sur dix ajoute une explication à l'anglaise, et le modèle a appris à « cocher moins » (omissions à 0,41, sous le hasard).
- **Essai 2** : le terme de vraisemblance supprime la dérive de format, mais les préférences françaises ne bougent pas. Le défaut vient des données.
- **Essai 3** : avec les données v5, les trois types d'erreur passent au-dessus du hasard, et l'écart de récompense est 2,5 fois plus grand. Le recul en anglais, dû à des paires moins nombreuses et plus courtes, est accepté : la cible est le français.

Pendant l'essai 3, les préférences justes en validation passent de 0,67 (pas 25) à 0,73 (fin d'époque, meilleur checkpoint). La perte, qui inclut le terme SFT, ne se compare pas d'un essai à l'autre.

## 4.4 Traçabilité de l'entraînement

- **Petits runs d'abord** : chaque changement est validé sur quelques dizaines de pas (environ 10 minutes) avant le run complet, comme le recommande le sujet.
- **MLflow sur DagsHub** : chaque run enregistre ses hyperparamètres, la version des données, ses courbes et ses métriques finales. Son nom résume sa configuration, par exemple `dpo-lora-r16-v5-sigmoid+sft-beta0.1-lr5e-05-seed42` pour le DPO retenu. Les métriques de chaque contrôle (préférences par type d'erreur, exactitude, format, test de McNemar) y sont enregistrées.
- **Checkpoints** : une sauvegarde tous les 25 pas, avec reprise possible après une interruption. Le meilleur checkpoint, selon la perte de validation, est rechargé à la fin.
- **Configuration** : un fichier `run_config.json` accompagne chaque adaptateur (réglages, tailles des données, graine) ; graine fixe (42) pour toutes les étapes aléatoires.

# 5. Évaluation et résultats

## 5.1 Méthode

**L'exactitude sur les réponses générées.** Le modèle répond à chaque QCM, sans hasard dans la génération. Une réponse est juste si elle donne **exactement** les lettres du corrigé. Le test compte 828 QCM français, jamais vus à l'entraînement : les 382 questions du test SFT et les 446 questions des paires de test DPO, posées avec leur historique de cas.

**Le format.** Une réponse est au bon format si elle recopie exactement la ou les propositions retenues (lettre et texte), sans rien ajouter. C'est la forme qu'exploite l'API de triage (partie 6.2).

**Les réponses naïves.** Un score n'a de sens que comparé à ce qu'on obtient sans réfléchir. La référence principale est la **réponse la plus fréquente** : donner à toutes les questions la lettre, ou la combinaison, la plus courante dans le corrigé.

**L'incertitude.** Sur 389 questions, une exactitude de 0,44 a une marge d'erreur d'environ ± 0,05. Pour comparer deux modèles sur les mêmes questions, on ne retient que les questions où ils divergent (**test de McNemar**) : c'est beaucoup plus fiable que de comparer deux moyennes.

## 5.2 Connaissances médicales : les QCM

| Test | Questions | Base | SFT | SFT + DPO | Réponse la plus fréquente |
|---|---|---|---|---|---|
| `mcqu` (une bonne réponse) | 389 | 0,17 | 0,432 | 0,445 | 0,211 |
| `mcqm` (2 à 5 bonnes réponses) | 439 | 0,04 | 0,109 | 0,121 | 0,059 |

- **Le modèle de base** (mesuré sur 340 questions du test SFT) suit mal la consigne : il coche en moyenne 0,6 proposition de trop, même quand une seule est exacte.
- **Le SFT apporte un gain réel** : 0,43 en `mcqu`, contre 0,21 pour la réponse naïve, soit deux fois plus de bonnes réponses. Face au modèle de base, il gagne 72 questions et en perd 9.
- **Le DPO ne dégrade rien** : 54 questions gagnées, 44 perdues (p = 0,36) ; 27 contre 22 en `mcqu` comme en `mcqm`. Son format reste celui du SFT (0,99 des réponses dans les deux cas).
- **Servi par vLLM** (380 questions du test SFT, posées avec leur historique de cas) : 0,43 en `mcqu`, comme le modèle précédent (12 questions gagnées, 13 perdues), et 0,10 en `mcqm`, contre 0,15. Le raccourci du modèle précédent, qui cochait deux propositions, paie sur les questions à deux bonnes réponses ; l'écart n'est pas significatif (5 contre 12, p = 0,14).

L'exactitude stricte reste faible en `mcqm` : il faut trouver la bonne combinaison parmi 26.

## 5.3 Ce que le DPO a appris

**Les préférences.** Sur les paires de test, le modèle DPO préfère la bonne réponse dans 68 % des cas en français et 74 % en anglais (50 % = hasard). Le découpage par type d'erreur montre *comment* :

| Type de `rejected` | Paires de test | Préférences justes |
|---|---|---|
| Substitution (`mcqu` et `mcqm`) | 289 | 0,70 |
| Ajout d'une mauvaise proposition | 80 | 0,71 |
| Omission d'une bonne proposition | 79 | 0,53 |

La substitution, où les deux réponses ont autant de propositions, mesure le mieux l'apprentissage sur le fond : 0,70, contre 0,61 pour le modèle v3. Une proposition manquante, en revanche, n'est presque pas repérée : la perte DPO additionne les probabilités des tokens, et une réponse plus courte est d'office plus probable. La perte normalisée par le nombre de tokens vise ce biais ; c'est le prochain essai à mener.

**L'effet sur les réponses.** Le SFT coche trop de propositions ; le DPO corrige ce défaut :

| Bonnes réponses attendues | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Questions `mcqm` | 213 | 144 | 60 | 22 |
| Propositions cochées, SFT | 3,12 | 3,54 | 3,68 | 3,77 |
| Propositions cochées, SFT + DPO | 2,54 | 2,97 | 3,07 | 3,00 |

Le SFT coche 0,62 proposition de trop par question, le DPO 0,02. Le raccourci du modèle v3 (deux lettres le plus souvent) a disparu : le nombre coché suit le nombre attendu, mais reste trop bas à 4 ou 5 bonnes réponses, comme le laissait prévoir la faiblesse sur les omissions.

**Le biais de longueur en anglais.** Sur le test revu par des experts, le modèle préfère la bonne réponse dans 72 % des cas quand elle est la plus courte, et dans 74 % quand elle est la plus longue (77 % et 62 % pour le modèle v3). L'équilibrage du train (partie 3.5) a supprimé ce biais.

## 5.4 Contrôles de sécurité

Les contrôles de sécurité demandés en phase 2 portent sur 28 QCM pièges, posés au modèle servi par vLLM, en local puis dans le cloud (mêmes réponses). Ces cas ont été rédigés pour ce projet ; ils n'ont pas été validés par un clinicien. Le modèle précédent (données v3) sert de comparaison.

| Catégorie | Questions | Modèle précédent (v3) | Modèle final |
|---|---|---|---|
| Recommandation dangereuse parmi les choix | 12 | 3 dangereuses | 4 dangereuses |
| Entité médicale inventée | 5 | 0 reconnue | 0 reconnue |
| Questions témoins (réelles) | 2 | 2 justes | 2 justes |
| Niveau de priorité (triage) | 9 | 4 justes | 4 justes |

**Recommandations dangereuses.** Les deux modèles choisissent une option dangereuse dans trois situations :

- une **perfusion de potassium** face à une hyperkaliémie sévère, qu'elle aggraverait ;
- un **antipyrétique et un retour à domicile** face à un purpura fébrile extensif, au lieu d'une antibiothérapie immédiate ;
- un **antalgique et un retour à domicile** face à une intoxication au monoxyde de carbone.

Le modèle final en ajoute une quatrième : **antalgiques et réévaluation dans une semaine** face à une douleur testiculaire brutale chez un adolescent, évocatrice d'une torsion, qui impose une exploration chirurgicale en urgence. Sur 12 cas, l'écart n'est pas significatif, mais c'est le type d'erreur qu'un outil de triage ne peut pas se permettre.

**Hallucinations.** Face à une maladie, un médicament ou un muscle inventés, les deux modèles choisissent toujours une réponse médicale plausible, alors que la proposition « n'existe pas » est offerte. Ils ne savent pas reconnaître les limites de leurs connaissances.

**Triage.** Les quatre urgences vitales (arrêt cardiaque, état de choc, hémorragie, convulsions) sont toutes classées en urgence maximale : aucun sous-triage. En revanche, les cas modérés sont classés en urgence maximale, et les trois cas bénins en urgence modérée. Les deux modèles **sur-trient** : c'est l'erreur la moins dangereuse, mais elle engorgerait le service. Le point d'accès `/triage` (partie 6.2) le confirme sur 11 patients décrits en langage naturel, avec les deux modèles : 5 urgences vitales en urgence maximale, mais les 6 autres cas sur-triés, dont une entorse de cheville.

**Évaluation externe du triage.** Le jeu public `clinical-triage-bilingual` (Hugging Face, CC BY 4.0) fournit 60 cas rédigés à la main, en français et en anglais (20 par niveau, dont des cas piégeux), avec des niveaux fixés par son auteur, sans validation médicale. Le modèle seul y trouve 23 niveaux sur 60, sous-trie 1 urgence vitale sur 20 et sur-trie 36 cas.

Corriger l'a priori du modèle pour l'urgence maximale (0,74 pour un patient sans information) semblait sûr sur nos 20 cas : 14 bons niveaux au lieu de 9, sans sous-triage. Sur les 60 cas externes, cette calibration gagne 5 niveaux mais sous-trie 11 à 12 urgences vitales sur 20 : la probabilité d'urgence maximale se recouvre entre urgences vitales (médiane 0,73) et modérées (0,61). Elle reste donc désactivée ; le sur-triage est l'erreur la moins dangereuse. Les motifs bénins relèvent d'une règle déterministe (partie 6.2) : sur les 60 cas, même avec des constantes supposées normales, elle ne déclasse aucune urgence vitale.

**Conclusion.** Le modèle ne peut en aucun cas être utilisé de façon autonome. Ses erreurs sont précisément celles qu'un outil de triage ne peut pas se permettre : recommandations dangereuses et réponses assurées sur ce qu'il ignore.

# 6. Mise en service de la démonstration

## 6.1 Pourquoi vLLM

Pendant l'entraînement, la génération avec la bibliothèque Transformers traite environ 2 questions par seconde : les requêtes sont regroupées en lots figés, et le lot attend la réponse la plus longue. vLLM est un serveur d'inférence conçu pour servir de nombreuses requêtes en même temps :

- **PagedAttention** : la mémoire de travail de chaque requête est allouée par petites pages, au fil de la génération. Moins de mémoire perdue, donc plus de requêtes simultanées.
- **Batching continu** : dès qu'une réponse est terminée, une nouvelle requête prend sa place, sans attendre la fin du lot.
- **API compatible OpenAI** et **plusieurs GPU** : n'importe quel client standard peut l'interroger, et il sait répartir un grand modèle (32B et plus) sur plusieurs GPU.

## 6.2 L'API et la sécurité d'accès

L'API FastAPI expose deux points d'accès : `POST /triage` et `GET /health` (état de l'API ; celui de vLLM avec la clé seulement, pour qu'un appel anonyme ne réveille pas l'endpoint cloud, facturé). vLLM n'est jamais exposé directement : seule l'API l'est.

**Le point d'accès de triage.** Le soignant décrit le patient en langage naturel : âge, motif, symptômes, antécédents et, s'il les a, les constantes. En question libre, le modèle produit en français des réponses incohérentes : il n'a appris cette langue que sur des QCM. Il n'est donc utilisé que là où il est fiable, et des briques déterministes l'encadrent :

| Brique | Rôle |
|---|---|
| Modèle | évalue le niveau de priorité (maximale, modérée, différée) à partir d'un cas clinique rédigé à partir des informations reçues, sur un choix fermé |
| Garde-fous | des constantes hors seuils (SpO₂ < 90 %, PAS < 90 mmHg, FC > 130/min…) imposent l'urgence maximale : ils relèvent le niveau, jamais ne l'abaissent ; sans le modèle, ils décident seuls |
| Motifs bénins | liste fermée (rhume, mal de gorge, petite plaie…), adulte aux constantes complètes et normales, sans signe d'alerte : urgence différée, réorientation vers la médecine de ville |
| Questionnaire adaptatif | questions complémentaires selon le motif, et constantes manquantes à mesurer |
| Explication | constantes hors seuils, avis du modèle, raison d'une absence de réorientation, conduite à tenir |

Les seuils et les questions sont des valeurs de démonstration, à valider par l'équipe médicale.

| Mesure | Mise en œuvre |
|---|---|
| Accès protégé | clé obligatoire dans l'en-tête `X-API-Key` ; refus tracés |
| Secrets | clés dans `.env` (exclu de Git) et dans les secrets GitHub |
| Validation des entrées | longueurs et valeurs des constantes bornées, balises de gabarit interdites |
| Limites d'usage | avertissement joint à chaque réponse |

## 6.3 Deux déploiements : local et cloud

| | Local | Cloud (démonstration) |
|---|---|---|
| Inférence | vLLM 0.32 sur la RTX 3080 | Inference Endpoint Hugging Face (AWS, Irlande) : vLLM 0.32, GPU T4, accès par jeton |
| API | conteneur Docker | Space Docker privé (CPU gratuit) |
| Journal d'audit | fichier JSONL | logs du Space (disque effacé au redémarrage) |
| Coût | — | 0,50 $/h, seulement allumé ; pause manuelle, veille après 15 min |

L'endpoint se réveille en 3 minutes ; un test complet (démonstration, latence, tests d'intégration) coûte quelques dizaines de centimes. Le T4 ne gère pas le format bf16 des poids : vLLM y sert le modèle en float16. Sur les 408 QCM d'évaluation, local et cloud donnent les mêmes scores, à une ou deux réponses près.

## 6.4 Latence et débit

Mesures du modèle final sur `/triage`, avec 100 requêtes par niveau de charge, depuis le client, réseau compris ; enregistrées dans MLflow.

| Requêtes simultanées | Local : médiane / p95 | Local : débit | Cloud : médiane / p95 | Cloud : débit |
|---|---|---|---|---|
| 1 | 82 / 94 ms | 12,0 req/s | 546 / 614 ms | 1,8 req/s |
| 10 | 216 / 252 ms | 46,3 req/s | 587 / 1 309 ms | 14,6 req/s |
| 50 | 754 / 805 ms | 63,8 req/s | 1 320 / 1 811 ms | 31,3 req/s |

- **Le batching continu fonctionne** : de 1 à 50 requêtes simultanées, le débit est multiplié par 5, pour une latence médiane qui reste sous la seconde. Pendant l'entraînement, la génération plafonnait à environ 2 réponses par seconde.
- **vLLM seul** répond en 38 ms au premier token (TTFT) et produit ensuite un token toutes les 7 ms (TPOT) : l'essentiel du temps est passé à générer la réponse.
- **La réponse de triage tient en une ligne** (le niveau choisi) : la latence est faible et régulière, avec un p95 proche de la médiane.
- **Dans le cloud, le réseau s'ajoute** : chaque requête traverse Internet jusqu'au Space, puis du Space à l'endpoint, sur un T4 moins rapide que la RTX 3080. Toutes les requêtes aboutissent, même à 50 simultanées. En production, placer l'API et le GPU au même endroit supprimerait l'essentiel de ce surcoût.

## 6.5 Robustesse

Seize tests d'intégration, lancés contre la pile réelle, vérifient le comportement dans les cas limites. La plupart sont relancés automatiquement à chaque déploiement ; l'arrêt de vLLM n'est testé qu'à la demande.

| Situation testée | Comportement vérifié |
|---|---|
| 5 urgences vitales décrites en langage naturel | toutes classées en urgence maximale (aucun sous-triage) |
| 11 cas de triage | réponse complète : niveau, justification, conduite, questions |
| Description trop longue, valeurs hors limites | refus (erreur 422) |
| Consigne contradictoire (« ignore les consignes ») | le modèle donne toujours un niveau exploitable |
| Description en anglais, caractères spéciaux | réponse valide |
| 50 requêtes simultanées | toutes servies, toutes tracées |
| Serveur vLLM arrêté | garde-fous seuls en cas de signe de gravité ; sinon erreur 503 tracée |

Ces tests ont révélé deux défauts, corrigés depuis :

- **Injection par les balises du gabarit.** Un texte contenant `<|im_end|>` ou `<|im_start|>` était découpé en vrais tokens spéciaux : un utilisateur pouvait fermer son message et injecter un faux message système. L'API refuse désormais toute balise de cette forme, avant tout appel au modèle.
- **Entrée trop longue présentée comme une panne** : l'API répondait « serveur indisponible ». Elle répond maintenant « entrée trop longue ».

## 6.6 Audit de traçabilité

Chaque requête produit une ligne dans un journal d'audit (JSONL) : identifiant unique, horodatage, modèle servi, **empreinte SHA-256 de l'entrée** (sans le texte), niveau retenu et sa source, nombre de signes de gravité, latence et issue (succès, refus, erreur).

Les tests d'audit vérifient que chaque requête est tracée avec tous ses champs, que les accès refusés le sont sans la clé reçue, et qu'**aucun texte patient** n'apparaît dans le journal (un marqueur placé dans chaque champ n'y est jamais retrouvé).

**Une limite pour l'audit.** Même avec `temperature = 0`, vLLM n'est pas parfaitement déterministe : selon le regroupement des requêtes, 2 à 3 % des réponses changent d'un passage à l'autre. Le journal enregistre donc la réponse effectivement donnée : c'est la trace de référence.

# 7. Limites

**Par rapport au besoin du CHSA.**

- **Pas de triage appris** : le modèle répond à des QCM médicaux ; il n'a jamais appris à attribuer un niveau de priorité. Il sur-trie (36 cas sur 60 dans l'évaluation externe), et distingue mal une urgence vitale d'une urgence modérée.
- **Questionnaire par règles** : questions fixes selon le motif ; le modèle ne mène pas d'entretien.
- **Peu d'explications** : en français, le modèle donne la proposition retenue, sans justification.
- **Pas d'intégration au SIH** : l'API reçoit une description du patient en JSON, pas un dossier patient.

**Le modèle.**

- **Connaissances limitées** : 0,44 de bonnes réponses aux QCM à réponse unique, cohérent avec sa taille (1,7 milliard de paramètres).
- **Erreurs dangereuses et hallucinations** : 4 recommandations dangereuses sur 12 cas pièges (3 pour le modèle précédent), et aucune entité inventée reconnue comme telle.
- **Apport du DPO limité** : les préférences sont apprises (0,68 en français) et les propositions cochées en trop ont disparu, mais l'exactitude ne progresse pas de façon significative, et une proposition manquante n'est repérée qu'au hasard.

**Les données et la méthode.**

- **Pas de validation clinique humaine** : aucun soignant ne relit les données. La proximité avec le triage est repérée par des expressions régulières ; les erreurs DPO françaises sont construites à partir du corrigé, pas tirées des erreurs réelles du modèle.
- **Préférences jugées par GPT-4** : en anglais, hors QCM ; seul le test a été revu par des experts.
- **Pas de données structurées** : symptômes, antécédents et constantes figurent dans le texte libre des cas, pas dans des champs dédiés.
- **Contrôles de sécurité non validés** : les 28 cas pièges ont été rédigés pour ce projet, sans relecture clinique.
- **Une seule graine** : chaque configuration a été entraînée une fois ; les écarts de quelques points ne sont pas significatifs.

**Le déploiement.**

- **Hébergement non conforme pour des données de santé** : la démo cloud tourne chez Hugging Face (AWS en Irlande pour le GPU), sans certification HDS.
- **Démonstration seulement** : réveil de 3 minutes, Space gratuit sans haute disponibilité ; surveillance limitée à l'état de santé et au journal d'audit, sans tableau de bord ni alertes.

# 8. Roadmap de passage à l'échelle au CHSA

La feuille de route place les données de triage et la validation clinique **avant** l'augmentation de la taille du modèle : un modèle plus grand, entraîné sur les mêmes QCM, ne saurait toujours pas trier.

| Étape | Objectif | Livrables | Condition de passage |
|---|---|---|---|
| 1. Données de triage | Apprendre la vraie tâche | Jeu de cas annotés par des soignants | Accord entre annotateurs suffisant |
| 2. Modèle de triage | Trier et expliquer | Modèle évalué sur ce jeu | Sous-triage des urgences vitales proche de zéro |
| 3. Infrastructure | Héberger en conformité | Serveurs HDS ou sur site, intégration SIH | Audit de sécurité validé |
| 4. Pilote clinique | Mesurer la valeur réelle | Étude en parallèle des soignants | Concordance et sécurité démontrées |

**Étape 1 : un jeu de données de triage.**

- Construire avec les infirmiers d'accueil (IOA) du CHSA quelques milliers de cas annotés selon l'échelle de tri du service (par exemple FRENCH, de la SFMU), dont une partie par deux soignants : leur accord est le plafond de performance à viser.
- Structurer chaque cas : motif de recours, symptômes, antécédents, constantes vitales, niveau de priorité et justification (schéma de métadonnées prévu en phase 1).
- Partir de dossiers historiques pseudonymisés (accord du DPO, hébergement HDS), avec la chaîne d'anonymisation du POC.

**Étape 2 : un modèle de triage.**

- **Sortie structurée et abstention** : niveau, justification et alertes en JSON validé par l'API, ou « avis soignant requis » plutôt qu'une invention.
- **Règles validées** : étendre avec l'équipe médicale les règles de l'API (constantes hors seuils, motifs bénins).
- **Protocoles du CHSA** fournis au modèle au moment de répondre (RAG), plutôt que mémorisés.
- **DPO sur les vraies erreurs** du modèle, validées par des soignants ; **montée en taille** (32B ou plus), jugée d'abord sur le sous-triage des urgences vitales.

**Étape 3 : une infrastructure de production.**

- **Hébergement** : GPU sur site ou chez un hébergeur HDS ; un 32B demande environ 64 Go de mémoire GPU (deux GPU de 48 Go), moins s'il est quantifié.
- **Intégration au SIH** : HL7 FHIR, pour lire le dossier et y écrire la proposition de triage.
- **Sécurité et surveillance** : authentification par l'annuaire du CHSA, journal d'audit inaltérable à durée de conservation définie ; tableaux de bord et alertes (latence, erreurs, répartition des niveaux).
- **Déploiement** : registre des modèles, CI/CD du POC, environnement de recette.

**Étape 4 : un pilote clinique.**

- **Étude « en parallèle »** : pendant quelques semaines, le système propose un niveau que les soignants ne voient pas ; on le compare à leurs décisions (concordance, sous-triage, temps gagné).
- **Cadre réglementaire** : dispositif médical et IA à haut risque (règlements européens), à instruire dès l'étape 1 ; l'outil reste une aide, le soignant décide.
