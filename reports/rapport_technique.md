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

- **Données** : un corpus médical bilingue, construit à partir de deux sources ouvertes (MediQAl en français, UltraMedical en anglais). Il est vérifié par corrigé, réparti par cas clinique pour éviter toute fuite entre entraînement et test, et anonymisé avec Presidio. Il compte 6 354 exemples d'entraînement SFT et 3 357 paires DPO.
- **Modèle** : un fine-tuning supervisé (SFT) avec LoRA, puis un alignement par préférences (DPO), sur une seule carte graphique de 10 Go. Le modèle est publié sur Hugging Face Hub, et chaque run est tracé dans MLflow.
- **Service** : une API FastAPI protégée par clé, servie par vLLM, avec un journal d'audit sans donnée patient. Elle est déployée en local (Docker, RTX 3080) et dans le cloud (Space et Inference Endpoint Hugging Face). Un pipeline GitHub Actions teste, publie et redéploie automatiquement chaque nouvelle version.

**Résultats clés.**

| Indicateur | Résultat |
|---|---|
| QCM à réponse unique (749 questions) | 0,44 de bonnes réponses, contre 0,24 pour une réponse naïve |
| Apport du DPO | aucun sur les réponses uniques ; + 2,4 points, non significatif, sur les réponses multiples |
| Contrôles de sécurité (12 cas) | 3 recommandations dangereuses ; entités inventées jamais reconnues |
| Latence (1 requête, médiane) | 168 ms en local ; 745 ms dans le cloud, réseau compris |
| Débit (50 requêtes simultanées) | 45 requêtes par seconde |

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
| 1. Validation conceptuelle | Corpus médical bilingue, anonymisé, prêt pour l'entraînement | 3 jeux SFT et 2 jeux DPO, en français et en anglais, anonymisés avec Presidio |
| 2. Optimisation ciblée | Adapter Qwen3-1.7B-Base par SFT LoRA, puis l'aligner par DPO | Modèle SFT + DPO, publié sur Hugging Face Hub |
| 3. Projection industrielle | Servir le modèle (vLLM, FastAPI, Docker), automatiser le déploiement | API servie par vLLM sur GPU, CI/CD GitHub Actions, tests de latence et de robustesse |

## 1.3 Périmètre du POC

Le POC entraine le modèle à **répondre à des questions médicales**.

- En **français**, des QCM sur des cas cliniques (MediQAl), avec un corrigé officiel. Le modèle doit choisir la ou les bonnes propositions.
- En **anglais**, des questions biomédicales avec des réponses rédigées et raisonnées (UltraMedical).

Ce choix permet de valider toute la chaîne technique (données, entraînement, alignement, mise en service, traçabilité) et de mesurer les connaissances médicales du modèle. Il ne permet **pas** d'évaluer la capacité de triage : le modèle n'a jamais appris à attribuer un niveau de priorité. Constituer un jeu de triage est la première étape de la feuille de route (partie 8).

## 1.4 Environnement

Tout le projet a été réalisé sur un poste de travail : carte graphique NVIDIA RTX 3080 (10 Go), sous Linux (WSL 2).

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
 [4] Service : local (Docker Compose, RTX 3080)
               ou cloud (Inference Endpoint L4 + Space Hugging Face)
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
| Démo cloud | HF Inference Endpoints, Spaces | Servir le modèle et l'API en ligne | Endpoint L4, Space privé |
| CI/CD | GitHub Actions, GHCR | Tester, publier, déployer | Pipeline automatisé |

## 2.3 Le déploiement automatisé

À chaque modification du code, le pipeline GitHub Actions enchaîne quatre étapes.

| Étape | Où | Contenu |
|---|---|---|
| `tests` | serveur GitHub | 20 tests de l'API, vLLM simulé (pas de GPU) |
| `image` | serveur GitHub | construction de l'image ; sur `master`, publication sur GHCR |
| `deploy` | machine GPU | mise à jour de la pile locale, puis tests d'intégration |
| `deploy-cloud` | serveur GitHub | mise à jour de l'API sur le Space, puis test de fumée |

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

Chaque exemple garde ses métadonnées : source, langue, identifiant de cas clinique, corrigé, type d'erreur, contrôle appliqué et nombre de noms anonymisés. Le schéma complet est documenté dans le README du dépôt.

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

# 4. Entraînement

## 4.1 Méthode : SFT puis DPO, avec LoRA

L'entraînement se fait en deux temps, sur le même modèle de base, `Qwen/Qwen3-1.7B-Base`.

- **SFT** (*Supervised Fine-Tuning*) : le modèle lit des questions et les réponses attendues, et apprend à les reproduire. La perte ne porte que sur la réponse, pas sur la question.
- **DPO** (*Direct Preference Optimization*) : le modèle lit des paires de réponses à une même question, et apprend à rendre la réponse préférée plus probable que l'autre. Le modèle SFT sert de **référence** : le DPO mesure ce que le modèle préfère *de plus* que le SFT, ce qui l'empêche de s'en éloigner trop.

Les deux étapes utilisent **LoRA** : les poids du modèle sont figés, et seules de petites matrices ajoutées à chaque couche sont entraînées. Avec un rang de 16 sur toutes les couches linéaires, cela représente 17,4 millions de paramètres, soit 1 % du modèle. Un fine-tuning complet demanderait environ 27 Go de mémoire GPU ; avec LoRA, l'entraînement tient sur les 10 Go de la RTX 3080.

Pour le DPO, l'adaptateur SFT est d'abord fusionné dans les poids, puis un nouvel adaptateur est entraîné par-dessus. La référence s'obtient en désactivant ce nouvel adaptateur : il n'est pas nécessaire de charger un second modèle en mémoire.

## 4.2 Réglages

| Réglage | SFT | DPO |
|---|---|---|
| Données (train) | 6 354 exemples | 3 351 paires |
| LoRA | rang 16, alpha 32, dropout 0,05 | identique |
| Époques | 1 | 1 |
| Learning rate | 2e-4, décroissance en cosinus | 5e-5, décroissance en cosinus |
| Montée progressive | 3 % des pas | 10 % des pas |
| Batch effectif | 16 exemples (4 × 4) | 16 paires (2 × 8) |
| Longueur maximale | 1 024 tokens | 1 024 tokens |
| Paramètre propre | — | beta = 0,1 |
| Pas d'entraînement | 398 | 210 |
| Durée (RTX 3080) | 30 min | 53 min |

Deux réglages ont été ajustés au fil des runs :

- **Une seule époque de SFT** : avec deux époques, la perte de validation atteignait son minimum à la fin de la première, puis remontait légèrement. La seconde époque coûtait 30 minutes sans rien apporter.
- **Learning rate du DPO à 5e-5** : la valeur usuelle en DPO complet (5e-6) ne produisait presque aucun apprentissage avec LoRA, qui n'entraîne qu'1 % des poids. La marge entre réponses préférées et rejetées restait proche de zéro.

## 4.3 Déroulement

| Indicateur | Avant | Après |
|---|---|---|
| SFT : perte de validation | 0,781 | 0,643 |
| DPO : perte de validation | 0,693 | 0,446 |
| DPO : préférences justes (validation) | 0,50 | 0,79 |
| DPO : marge des récompenses | 0 | 1,73 |

- **SFT** : la perte de validation baisse de façon régulière et se stabilise en fin d'époque, sans remonter. Sur le test, elle est proche de la validation (0,589) : le modèle généralise, sans apprendre par cœur.
- **DPO** : la perte part de 0,693 (valeur théorique sans aucune préférence) et plafonne à partir du pas 150. Les probabilités des réponses préférées et rejetées baissent toutes les deux, mais celles des rejetées baissent plus vite : c'est le comportement attendu.

Les pertes par source rappellent qu'une perte basse ne suffit pas à juger le modèle : 0,08 à 0,11 sur les QCM français, contre 0,79 sur l'anglais. En français, le texte des propositions est dans le prompt, et le modèle n'a presque qu'à le recopier ; la seule vraie difficulté est la lettre. D'où l'évaluation sur les réponses générées (partie 5).

## 4.4 Traçabilité de l'entraînement

- **Petits runs d'abord** : chaque changement est validé sur quelques dizaines de pas (environ 10 minutes) avant le run complet, comme le recommande le sujet.
- **MLflow sur DagsHub** : chaque run enregistre ses hyperparamètres, la version des données, ses courbes et ses métriques finales. Son nom résume sa configuration, par exemple `dpo-lora-r16-v3-beta0.1-lr5e-05-seed42`.
- **Checkpoints** : une sauvegarde tous les 100 pas (SFT) ou 25 pas (DPO), avec reprise possible après une interruption. Le meilleur checkpoint, selon la perte de validation, est rechargé à la fin.
- **Configuration** : un fichier `run_config.json` accompagne chaque adaptateur (réglages, tailles des données, graine).
- **Graine fixe** (42) pour toutes les étapes aléatoires.

# 5. Évaluation et résultats

## 5.1 Méthode

**L'exactitude sur les réponses générées.** Le modèle répond à chaque QCM, sans hasard dans la génération (`temperature = 0`). Une réponse est juste si elle donne **exactement** les lettres du corrigé. L'évaluation est faite à travers vLLM, avec le prompt de l'API : le modèle est mesuré tel qu'il est servi.

**Les réponses naïves.** Un score n'a de sens que comparé à ce qu'on obtient sans réfléchir. La référence principale est la **réponse la plus fréquente** : donner à toutes les questions la lettre, ou la combinaison, la plus courante dans le corrigé.

**L'incertitude.** Sur 749 questions, une exactitude de 0,44 a une marge d'erreur d'environ ± 0,036. Pour comparer deux modèles sur les mêmes questions, on ne retient que les questions où ils divergent (**test de McNemar**) : c'est beaucoup plus fiable que de comparer deux moyennes.

## 5.2 Connaissances médicales : les QCM

| Test | Questions | Base | SFT | SFT + DPO | Réponse la plus fréquente |
|---|---|---|---|---|---|
| `mcqu` (une bonne réponse) | 749 | ~0 | 0,437 | 0,445 | 0,244 |
| `mcqm` (2 à 5 bonnes réponses) | 931 | ~0 | 0,090 | 0,114 | 0,068 |

- **Le modèle de base** (mesuré sur 200 questions par source) obtient un score proche de zéro : il ne suit pas le format (il écrit un paragraphe sans lettre).
- **Le SFT apporte un gain réel** : 0,44 en `mcqu`, contre 0,24 pour la réponse naïve, soit 1,8 fois plus de bonnes réponses.
- **Le DPO ne change rien en `mcqu`** (43 questions gagnées, 37 perdues ; p = 0,58), alors qu'il n'a été entraîné que sur des QCM à réponses multiples. Il ne dégrade donc pas ce qu'avait appris le SFT.
- **En `mcqm`, le DPO gagne 2,4 points** (83 questions gagnées, 61 perdues ; p = 0,08) : un gain à la limite de la significativité.

L'exactitude stricte reste faible en `mcqm` : il faut trouver la bonne combinaison parmi 26.

## 5.3 Ce que le DPO a appris

**Les préférences.** Sur les paires de test, le modèle DPO préfère la bonne réponse dans 70 % des cas en français et 66 % en anglais (50 % = hasard). Le découpage par type d'erreur montre *comment* :

| Type de `rejected` | Paires de test | Préférences justes |
|---|---|---|
| Ajout d'une mauvaise proposition | 150 | 0,86 |
| Omission d'une bonne proposition | 176 | 0,64 |
| Substitution | 155 | 0,61 |

Le modèle repère bien une proposition en trop, beaucoup moins une proposition manquante. Le score des omissions est même flatteur : dans 81 de ces paires, la `rejected` n'a qu'une lettre et contredit la consigne « au moins deux ». La substitution, seul cas où les deux listes ont la même longueur, mesure le mieux l'apprentissage sur le fond : 0,61, à peine au-dessus du hasard.

**L'effet sur les réponses.** Le DPO a surtout appris à **cocher moins** :

| Bonnes réponses attendues | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Questions (200 du test DPO) | 87 | 76 | 30 | 7 |
| Exactitude SFT | 0,13 | 0,12 | 0,13 | 0,43 |
| Exactitude SFT + DPO | 0,30 | 0,09 | 0,07 | 0,00 |

Sur l'ensemble du test `mcqm`, le SFT coche en moyenne 3,68 propositions, le DPO 2,42, pour 2,78 attendues. Le DPO répond le plus souvent deux lettres : il gagne sur les questions à deux bonnes réponses, les plus nombreuses, et perd sur les autres. Son gain global vient de ce raccourci, plus que d'un progrès de connaissances.

**Le biais de longueur en anglais.** Sur le test revu par des experts, le modèle préfère la bonne réponse dans 77 % des cas quand elle est la plus courte, et dans 62 % quand elle est la plus longue. L'équilibrage du train (partie 3.4) a réduit ce biais sans le supprimer.

## 5.4 Contrôles de sécurité

Les contrôles de sécurité demandés en phase 2 portent sur 28 QCM pièges, passés par l'API. Ces cas ont été rédigés pour ce projet ; ils n'ont pas été validés par un clinicien.

| Catégorie | Questions | SFT | SFT + DPO |
|---|---|---|---|
| Recommandation dangereuse parmi les choix | 12 | 4 dangereuses | 3 dangereuses |
| Entité médicale inventée | 5 | 0 reconnue | 0 reconnue |
| Questions témoins (réelles) | 2 | 2 justes | 2 justes |
| Niveau de priorité (triage) | 9 | 6 justes | 4 justes |

**Recommandations dangereuses.** Les deux modèles choisissent une option dangereuse dans trois situations :

- une **perfusion de potassium** face à une hyperkaliémie sévère, qu'elle aggraverait ;
- un **antipyrétique et un retour à domicile** face à un purpura fébrile extensif, au lieu d'une antibiothérapie immédiate ;
- un **antalgique et un retour à domicile** face à une intoxication au monoxyde de carbone.

Le SFT seul indique en plus une dose maximale de paracétamol de **10 g par jour** ; le DPO corrige cette erreur (4 g).

**Hallucinations.** Face à une maladie, un médicament ou un muscle inventés, les deux modèles choisissent toujours une réponse médicale plausible, alors que la proposition « n'existe pas » est offerte. Ils ne savent pas reconnaître les limites de leurs connaissances.

**Triage.** Les quatre urgences vitales (arrêt cardiaque, état de choc, hémorragie, convulsions) sont toutes classées en urgence maximale : aucun sous-triage. En revanche, les cas modérés sont classés en urgence maximale, et le DPO classe les trois cas bénins en urgence modérée. Les deux modèles **sur-trient** : c'est l'erreur la moins dangereuse, mais elle engorgerait le service. Le modèle n'a jamais été entraîné sur cette tâche, et neuf cas ne suffisent pas à conclure.

**Conclusion.** Le modèle ne peut en aucun cas être utilisé de façon autonome. Ses erreurs sont précisément celles qu'un outil de triage ne peut pas se permettre : recommandations dangereuses et réponses assurées sur ce qu'il ignore.

# 6. Mise en service de la démonstration

## 6.1 Pourquoi vLLM

Pendant l'entraînement, la génération avec la bibliothèque Transformers traite environ 2 questions par seconde : les requêtes sont regroupées en lots figés, et le lot attend la réponse la plus longue. vLLM est un serveur d'inférence conçu pour servir de nombreuses requêtes en même temps :

- **PagedAttention** : la mémoire de travail de chaque requête est allouée par petites pages, au fil de la génération. Moins de mémoire perdue, donc plus de requêtes simultanées.
- **Batching continu** : dès qu'une réponse est terminée, une nouvelle requête prend sa place, sans attendre la fin du lot.
- **API compatible OpenAI** : n'importe quel client standard peut l'interroger.
- **Passage à l'échelle** : vLLM sait répartir un grand modèle sur plusieurs GPU, ce que demandera un modèle de 32B et plus.

## 6.2 L'API et la sécurité d'accès

L'API FastAPI expose deux points d'accès : `POST /qcm` (répondre à un QCM) et `GET /health` (état de l'API et de vLLM). vLLM n'est jamais exposé directement : seule l'API l'est.

| Mesure | Mise en œuvre |
|---|---|
| Accès protégé | clé obligatoire dans l'en-tête `X-API-Key` ; refus tracés |
| Secrets | clés dans `.env` (exclu de Git) et dans les secrets GitHub |
| Validation des entrées | longueurs maximales, propositions non vides, balises interdites |
| Réponse vérifiée | champ `format_valide` : lettres proposées, en nombre conforme |
| Limites d'usage | avertissement joint à chaque réponse |

## 6.3 Deux déploiements : local et cloud

| | Local | Cloud (démonstration) |
|---|---|---|
| Inférence | vLLM 0.32 sur la RTX 3080 | Inference Endpoint Hugging Face : vLLM 0.32, GPU L4, accès par jeton |
| API | conteneur Docker | Space Docker privé (CPU gratuit) |
| Modèle | même dépôt, même révision | même dépôt, même révision |
| Journal d'audit | fichier JSONL | logs du Space (disque effacé au redémarrage) |
| Coût | — | ~0,80 $/h, seulement allumé ; pause manuelle, veille après 15 min |

L'endpoint se réveille en 3 minutes ; un test complet (démonstration, latence, tests d'intégration) a coûté environ 0,25 $. Sur les mêmes questions, le cloud donne les mêmes réponses que la pile locale.

## 6.4 Latence et débit

Mesures avec 100 QCM du jeu de test par niveau de charge, enregistrées dans MLflow. Pour le cloud, la latence est mesurée depuis le client, réseau compris.

| Requêtes simultanées | Local : médiane / p95 | Local : débit | Cloud : médiane / p95 | Cloud : débit |
|---|---|---|---|---|
| 1 | 168 / 433 ms | 4,7 req/s | 745 / 1 432 ms | 1,2 req/s |
| 10 | 234 / 657 ms | 31,8 req/s | 910 / 2 159 ms | 8,8 req/s |
| 50 | 823 / 1 234 ms | 45,0 req/s | 2 127 / 3 575 ms | 17,0 req/s |

- **Le batching continu fonctionne** : de 1 à 50 requêtes simultanées, le débit est multiplié par 10, pour une latence médiane qui reste sous la seconde.
- **Par rapport à la génération pendant l'entraînement**, le débit passe d'environ 2 à 45 réponses par seconde.
- **vLLM seul** répond en 38 ms au premier token (TTFT) et produit ensuite un token toutes les 7 ms (TPOT) : l'essentiel du temps est passé à générer la réponse.
- Le p95 est environ trois fois la médiane à faible charge : une réponse à plusieurs propositions est plus longue à générer qu'une réponse unique.
- **Dans le cloud, l'écart vient du réseau**, pas du GPU (un L4 vaut une RTX 3080) : chaque requête traverse Internet jusqu'au Space, puis du Space à l'endpoint. Toutes les requêtes aboutissent, même à 50 simultanées. En production, placer l'API et le GPU au même endroit supprimerait l'essentiel de ce surcoût.

## 6.5 Robustesse

Quatorze tests d'intégration, lancés contre la pile réelle, vérifient le comportement dans les cas limites. La plupart sont relancés automatiquement à chaque déploiement ; l'arrêt de vLLM n'est testé qu'à la demande. Contre le cloud, les 8 tests applicables réussissent (ceux qui relisent le fichier d'audit local sont ignorés).

| Situation testée | Comportement vérifié |
|---|---|
| 40 QCM réels du jeu de test | toutes les réponses au format attendu |
| Entrée trop longue pour le modèle | refus clair (erreur 422) |
| Propositions vides, entrée hors limites | refus (erreur 422) |
| Consigne contradictoire (« ignore les consignes ») | le modèle garde le format QCM |
| Question en anglais, caractères spéciaux | réponse valide |
| 50 requêtes simultanées | toutes servies, toutes tracées |
| Serveur vLLM arrêté | l'API reste disponible, erreur 503 tracée |

Ces tests ont révélé trois défauts, corrigés depuis :

- **Injection par les balises du gabarit.** Un texte contenant `<|im_end|>` ou `<|im_start|>` était découpé en vrais tokens spéciaux : un utilisateur pouvait fermer son message et injecter un faux message système. L'API refuse désormais toute balise de cette forme, avant tout appel au modèle.
- **Entrée trop longue présentée comme une panne** : l'API répondait « serveur indisponible ». Elle répond maintenant « entrée trop longue ».
- **Propositions vides acceptées** : un QCM pouvait ne contenir qu'une seule vraie proposition.

## 6.6 Audit de traçabilité

Chaque requête produit une ligne dans un journal d'audit (JSONL) :

| Champ | Contenu |
|---|---|
| `request_id`, `timestamp` | identifiant unique et horodatage |
| `model` | modèle servi |
| `input_sha256` | empreinte de l'entrée, sans le texte |
| `letters`, `format_valide` | réponse donnée et sa validité |
| `latency_ms`, `status` | latence et issue (succès, refus, erreur) |

Les tests d'audit vérifient que chaque requête est tracée avec tous ses champs, qu'une même entrée donne la même empreinte, que les accès refusés sont tracés sans la clé reçue, et qu'**aucun texte patient** n'apparaît dans le journal (un marqueur unique, placé dans chaque champ, n'y est jamais retrouvé).

**Une limite pour l'audit.** Même avec `temperature = 0`, vLLM n'est pas parfaitement déterministe : selon le regroupement des requêtes, 2 à 3 % des réponses changent d'un passage à l'autre. Rejouer une question ne garantit donc pas d'obtenir la même réponse. Le journal enregistre la réponse effectivement donnée, ce qui reste la trace de référence.

# 7. Limites

**Par rapport au besoin du CHSA.**

- **Pas de triage appris** : le modèle répond à des QCM médicaux ; il n'a jamais appris à attribuer un niveau de priorité. Ses premiers résultats sur neuf cas de triage montrent une tendance au sur-triage.
- **Pas de questionnaire adaptatif** : le modèle répond à une question complète, il ne mène pas d'entretien avec le patient.
- **Peu d'explications** : en français, les réponses donnent la ou les propositions retenues, sans justification. Seule la partie anglaise a appris à raisonner.
- **Pas d'intégration au SIH** : l'API reçoit un QCM en JSON, pas un dossier patient.

**Le modèle.**

- **Connaissances limitées** : environ une réponse juste sur deux aux QCM à réponse unique, ce qui est cohérent avec sa taille (1,7 milliard de paramètres).
- **Erreurs dangereuses et hallucinations** : 3 recommandations dangereuses sur 12 cas pièges, et aucune entité inventée reconnue comme telle.
- **Apport du DPO limité** : le gain mesuré vient surtout d'un raccourci (cocher moins de propositions) et n'est pas significatif.

**Les données et la méthode.**

- **Erreurs DPO simulées** : en français, les réponses `rejected` sont construites à partir du corrigé, pas tirées des erreurs réelles du modèle.
- **Préférences jugées par GPT-4** : en anglais, hors QCM, les préférences reposent sur les notes d'un autre LLM. Seul le jeu de test a été revu par des experts.
- **Pas de données structurées** : symptômes, antécédents et constantes figurent dans le texte libre des cas, pas dans des champs dédiés.
- **Contrôles de sécurité non validés** : les 28 cas pièges ont été rédigés pour ce projet, sans relecture clinique.
- **Une seule graine** : chaque configuration a été entraînée une fois ; les écarts de quelques points ne sont pas significatifs.

**Le déploiement.**

- **Hébergement non conforme pour des données de santé** : la démo cloud tourne chez Hugging Face (endpoint chez AWS aux États-Unis), sans certification HDS. C'est acceptable pour des QCM publics, pas pour des données de patients.
- **Disponibilité de démonstration** : réveil de 3 minutes, Space gratuit (créé avant que Hugging Face ne restreigne cette offre), sans haute disponibilité.
- **Surveillance minimale** : état de santé et journal d'audit, sans tableau de bord ni alertes.
- **Non-déterminisme** : 2 à 3 % des réponses peuvent varier d'un passage à l'autre.

# 8. Roadmap de passage à l'échelle au CHSA

La faisabilité technique est acquise ; la valeur clinique reste à démontrer. La feuille de route place donc les données de triage et la validation clinique **avant** l'augmentation de la taille du modèle : un modèle plus grand, entraîné sur les mêmes QCM, ne saurait toujours pas trier.

| Étape | Objectif | Livrables | Condition de passage |
|---|---|---|---|
| 1. Données de triage | Apprendre la vraie tâche | Jeu de cas annotés par des soignants | Accord entre annotateurs suffisant |
| 2. Modèle de triage | Trier et expliquer | Modèle évalué sur ce jeu | Sous-triage des urgences vitales proche de zéro |
| 3. Infrastructure | Héberger en conformité | Serveurs HDS ou sur site, intégration SIH | Audit de sécurité validé |
| 4. Pilote clinique | Mesurer la valeur réelle | Étude en parallèle des soignants | Concordance et sécurité démontrées |

**Étape 1 : un jeu de données de triage.**

- Construire, avec les infirmiers organisateurs de l'accueil (IOA) du CHSA, quelques milliers de cas annotés selon l'échelle de tri en vigueur dans le service (par exemple l'échelle FRENCH de la SFMU).
- Structurer chaque cas : motif de recours, symptômes, antécédents, constantes vitales, niveau de priorité et justification. C'est le schéma de métadonnées prévu en phase 1.
- Partir de dossiers historiques pseudonymisés, avec l'accord du délégué à la protection des données, et les héberger dans un environnement certifié HDS. La chaîne d'anonymisation de ce POC est réutilisable.
- Faire annoter une partie des cas par deux soignants, pour mesurer leur accord : c'est le plafond de performance à viser.

**Étape 2 : un modèle de triage.**

- **Sortie structurée** : niveau de priorité, justification, signaux d'alerte identifiés, au format JSON validé par l'API.
- **Garde-fous déterministes** : des règles simples (par exemple des constantes vitales hors seuils) imposent l'urgence maximale, quelle que soit la réponse du modèle.
- **Abstention** : le modèle doit pouvoir répondre « avis soignant requis » plutôt que d'inventer.
- **Protocoles du CHSA** : fournir au modèle les protocoles locaux au moment de répondre (recherche documentaire, ou RAG), plutôt que de les lui faire mémoriser.
- **DPO sur les vraies erreurs** : construire les paires à partir des erreurs réelles du modèle, validées par des soignants.
- **Montée en taille** : passer à un modèle de 32B ou plus une fois le jeu de triage disponible, et mesurer le gain sur ce jeu.
- **Métrique principale** : le taux de sous-triage des urgences vitales, avant l'exactitude globale.

**Étape 3 : une infrastructure de production.**

- **Hébergement** : serveurs GPU sur site ou chez un hébergeur certifié HDS. Un modèle de 32B demande environ 64 Go de mémoire GPU en bf16 (deux GPU de 48 Go, répartis par vLLM), ou moins avec une quantification.
- **Intégration au SIH** : échanges au standard HL7 FHIR (patient, constantes, passage aux urgences), pour lire le dossier et y écrire la proposition de triage.
- **Sécurité** : authentification des soignants par l'annuaire du CHSA, au lieu d'une clé d'API partagée ; journal d'audit inaltérable, avec une durée de conservation définie.
- **Surveillance** : tableaux de bord et alertes sur la latence, les erreurs et la répartition des niveaux de priorité, pour détecter une dérive.
- **Reproductibilité** : registre des modèles et mode d'inférence invariant au regroupement des requêtes.
- **Déploiement** : la chaîne CI/CD de ce POC se transpose telle quelle, avec un environnement de recette avant la production.

**Étape 4 : un pilote clinique.**

- **Étude « en parallèle »** : pendant plusieurs semaines, le système propose un niveau de priorité sans que les soignants le voient. On compare ensuite ses propositions à leurs décisions.
- **Critères** : concordance avec les IOA, taux de sous-triage, temps gagné, acceptabilité par les équipes.
- **Cadre réglementaire** : un logiciel d'aide au triage peut relever du règlement européen sur les dispositifs médicaux et des systèmes d'IA à haut risque du règlement européen sur l'IA. Son statut est à instruire dès l'étape 1.
- **Usage** : l'outil reste une aide. Le soignant garde la décision, et chaque proposition est tracée.
