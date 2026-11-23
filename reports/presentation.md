# Présentation des livrables : plan et déroulé (15 minutes)

**Consigne** : faire la démonstration du POC, et démontrer la faisabilité technique et la valeur ajoutée clinique du système d'agent intelligent de triage.

**Principe** : pas de diapositives. On montre les cinq livrables du sujet directement à l'écran (terminal, dépôt, MLflow, Hugging Face, GitHub, rapport). On ouvre sur le service en marche, pour accrocher le jury, puis on remonte la chaîne pour montrer comment il a été construit.

**Outils** :

- `scripts/demo.sh` déroule les démonstrations du terminal, étape par étape (Entrée pour passer à la suivante). Une étape seule : `./scripts/demo.sh <numéro>`. Démo cloud : `API_URL=https://0biyohan-triage-api.hf.space ./scripts/demo.sh`.
- `scripts/ui.py`, une interface web (Gradio), si le jury veut tester un patient de son choix : `uv run --with gradio python scripts/ui.py`, puis http://127.0.0.1:7860 (cible locale ou cloud au choix).

## 1. Plan minuté

| Temps | Livrable | À l'écran | Étapes du script |
|---|---|---|---|
| 0:00 – 1:00 | Introduction | README du dépôt (haut de page) | — |
| 1:00 – 3:00 | 3. Endpoint : le résultat | Terminal | 1, 2, 3 |
| 3:00 – 6:00 | 1. Dataset bilingue anonymisé | README, notebooks, page Hugging Face du dataset | — |
| 6:00 – 8:30 | 2. Modèle SFT + DPO | MLflow (DagsHub), page Hugging Face du modèle | — |
| 8:30 – 11:00 | 3. Endpoint : cohérence et solidité | Terminal, MLflow (latence) | 4, 5, 6, 7 |
| 11:00 – 12:30 | 4. Pipeline CI/CD | GitHub Actions, GHCR | 8 (liens) |
| 12:30 – 14:30 | 5. Rapport, et valeur clinique | `reports/rapport.pdf` | — |
| 14:30 – 15:00 | Conclusion | — | — |

## 2. Déroulé détaillé

### 0:00 – 1:00 · Introduction

**À l'écran** : le README du dépôt sur GitHub.

**À dire** :

- Le CHSA : urgences saturées, effectifs de triage insuffisants, risque de ne pas repérer à temps un cas critique.
- L'objectif du POC : construire toute la chaîne d'un agent IA de triage (données, entraînement, alignement, service, déploiement, traçabilité) avec un modèle compact, Qwen3-1.7B, sur une seule carte graphique.
- L'annonce : « Je vais vous montrer les cinq livrables : le service d'abord, puis comment il a été construit. »

### 1:00 – 3:00 · Livrable 3 : l'endpoint, le résultat

**À l'écran** : le terminal, étapes 1 à 3 du script.

1. **Le service est en ligne** : `/health` répond `ok` pour l'API et pour vLLM.
2. **Un patient décrit en langage naturel** (homme de 58 ans, douleur thoracique constrictive irradiant au bras gauche, sueurs) : « Urgence maximale », avec la justification, la conduite à tenir et les **questions à poser** (irradiation de la douleur, température à prendre), en moins d'une demi-seconde. La durée, déjà donnée (« depuis 40 minutes »), n'est pas redemandée.
3. **Des constantes graves** (détresse respiratoire, SpO₂ 85 %, FR 34/min) : les **garde-fous** détectent les constantes hors seuils, qui imposent l'urgence maximale quel que soit l'avis du modèle.

**À dire** :

- Le soignant décrit le patient comme à l'accueil : motif, symptômes, constantes. L'API propose un niveau de priorité, l'explique, et dit quelles questions poser ensuite.
- Le modèle ne décide pas seul : il évalue le niveau sur un choix fermé ; des règles déterministes l'encadrent.
- **Phrase clé** : « Un niveau de priorité expliqué en moins d'une seconde, avec les questions à poser : c'est le résultat. Voyons maintenant comment il a été construit. »

### 3:00 – 6:00 · Livrable 1 : le dataset bilingue anonymisé

**À l'écran** :

1. Le README, sections « Sources des données et licences » et « Schéma des métadonnées ».
2. Le notebook `prep_SFT_MediQAl_oeq.ipynb` : l'exemple d'une ligne (un patient et ses questions dans l'ordre), puis la cellule « Sélection pour le triage ».
3. Le notebook `prep_DPO_MediQAl.ipynb` : la paire affichée (`chosen` et `rejected`) et le tableau des types d'erreur.
4. La page Hugging Face du dataset public [`0biyohan/triage-medical-bilingual`](https://huggingface.co/datasets/0biyohan/triage-medical-bilingual) : la fiche, le visualiseur, l'étiquette `v1.0`.
5. Un rapport d'anonymisation : `python3 -m json.tool data/v4/anonymized/sft_mediqal_oeq_train.report.json`

**À dire** :

- **Sources** : les jeux proposés par le sujet. MediQAl en français (cas cliniques rédigés par des enseignants : questions ouvertes et QCM avec corrigé) et UltraMedical en anglais (vignettes d'examen, questions de patients, test revu par des experts). MedQuAD et FrenchMedMCQA ont été écartés : aucun patient décrit. Aucun de ces jeux n'a de niveau de triage.
- **Proche du triage** : chaque exemple décrit un patient, en priorité avec des constantes et des signes d'urgence. Sur près de 30 000 questions françaises, 2 998 sont retenues.
- **Qualité plutôt que quantité** : chaque exemple est vérifié par le corrigé ou par la note. Par exemple, 111 541 paires anglaises d'origine pour 2 000 exemples SFT retenus.
- **Pas de fuite** : les splits officiels de MediQAl partagent des cas cliniques. La répartition est recalculée par cas : aucun cas n'est vu à la fois en entraînement et en test, ni à la fois en SFT et en DPO.
- **DPO français** : la mauvaise réponse est construite à partir des distracteurs des enseignants. La moitié sont des substitutions, avec autant de propositions que la bonne réponse : seul le fond les départage.
- **RGPD** : anonymisation Presidio ciblée sur les champs qui décrivent un patient. Un contrôle par sondage a montré que la moitié des remplacements des questions ouvertes étaient des termes médicaux (« <PERSON> coli ») : de nouvelles règles les écartent, 4 759 détections au total.
- **Phrase clé** : « 4 998 paires SFT et 3 712 paires DPO, orientées triage, vérifiées, sans fuite, anonymisées, et publiées en accès libre avec leur fiche. »

### 6:00 – 8:30 · Livrable 2 : le modèle SFT + DPO

**À l'écran** :

1. MLflow sur DagsHub : les runs `sft-lora-r16-v4-seed42` et `dpo-lora-r16-v5-sigmoid+sft-beta0.1-lr5e-05-seed42`, leurs paramètres et leurs courbes (`eval_loss`, `rewards/accuracies`).
2. La page Hugging Face du modèle public [`0biyohan/qwen3-1.7b-medical-triage-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-triage-fr) : la fiche, le lien vers le dataset, la révision servie (`4af91a7`).

**À dire** :

- **SFT puis DPO, avec LoRA** : 1 % des poids entraînés, ce qui tient sur une carte de 10 Go. Le SFT apprend à répondre ; le DPO apprend à préférer la bonne réponse.
- **Trois essais de DPO**, chacun corrigeant un défaut mesuré :
  - le premier ajoutait des explications à l'anglaise et avait appris à « cocher moins » ;
  - le deuxième ajoute à la perte un terme de vraisemblance de la bonne réponse : le format est réparé ;
  - le troisième, retenu, change les données (moitié de substitutions, réponses anglaises plus courtes).
- **Résultat** : préférences justes à 0,68 en français et 0,74 en anglais (0,5 = hasard) ; les propositions cochées en trop passent de 0,62 à 0,02 par question ; aucune régression (McNemar, p = 0,36).
- **Traçabilité** : chaque run garde ses hyperparamètres, sa graine et ses versions de données (SFT v4, DPO v5). Le modèle et le dataset sont publiés et liés sur Hugging Face, à des versions fixées.
- **Phrase clé** : « Chaque modèle est reproductible : on sait sur quelles données et avec quels réglages il a été entraîné, et on peut tout télécharger. »

### 8:30 – 11:00 · Livrable 3 : l'endpoint, cohérence et solidité

**À l'écran** : le terminal, étapes 4 à 7 du script, puis la latence dans MLflow (expérience `qwen3-1.7b-serving`).

4. **Deux situations bénignes** :
   - un rhume chez un adulte aux constantes normales : « Urgence différée », décidée par la **règle des motifs bénins**, avec réorientation vers le médecin traitant ;
   - une entorse de cheville, hors de cette liste : le modèle décide et sur-trie (« Urgence maximale »). La justification l'explique : « motif non reconnu comme bénin ». À annoncer avant de lancer.
5. **Sécurité** : sans clé, refus (401) ; avec de fausses balises `<|im_end|>`, refus (422) avant tout appel au modèle.
6. **Journal d'audit** : une ligne par requête, sans texte patient. La recherche du mot « constrictive » (tiré des symptômes) dans le journal donne 0 occurrence.
7. **Tests d'intégration en direct** : 15 tests réussis en quelques secondes, contre la pile réelle, dont les 5 urgences vitales jamais sous-triées.

**À dire** :

- **Le modèle sur-trie** : il n'a jamais appris le triage. C'est l'erreur la moins dangereuse, mais elle engorgerait l'accueil.
- **D'où deux règles symétriques**, comme un protocole d'accueil : des constantes hors seuils relèvent le niveau ; une liste fermée de motifs bénins, chez un adulte aux constantes complètes et normales, sans aucun signe d'alerte, l'abaisse.
- **Une correction rejetée** : corriger statistiquement l'a priori du modèle semblait marcher sur nos 20 cas. Sur 60 cas externes, elle sous-triait 11 à 12 urgences vitales sur 20 : abandonnée. Le sur-triage reste préférable au sous-triage.
- **La faille des balises** a été trouvée par nos tests de robustesse, puis corrigée : un utilisateur pouvait injecter un faux message système.
- **Latence** : 82 ms en médiane en local, 64 requêtes par seconde avec 50 utilisateurs simultanés, sur une seule carte graphique ; 0,55 s dans le cloud, réseau compris.
- **Phrase clé** : « Cohérent sur les cas simples, prudent sur les autres, sécurisé, et tracé sans aucune donnée de santé. »

### 11:00 – 12:30 · Livrable 4 : le pipeline CI/CD

**À l'écran** :

1. GitHub Actions : une exécution réussie, avec ses quatre jobs (`tests`, `image`, `deploy`, `deploy-cloud`).
2. Le job `deploy`, exécuté sur le runner `p14-gpu` (la machine GPU), puis `deploy-cloud` (Space Hugging Face).
3. L'image de l'API sur GHCR, étiquetée par commit.

**À dire** :

- À chaque commit : 42 tests unitaires, construction de l'image et publication. Sur `master` : déploiement sur la machine GPU avec tests d'intégration, puis mise à jour de l'API dans le cloud avec un test de fumée.
- Le cloud : l'API sur un Space Hugging Face, le modèle servi par vLLM sur un Inference Endpoint (GPU T4, 0,50 $/h, mis en pause hors démonstration).
- Le déploiement remplace l'API en quelques secondes, sans redémarrer vLLM. Revenir à une version précédente, c'est relancer le pipeline d'un ancien commit.
- Les secrets (clé d'API, jetons Hugging Face) sont dans GitHub, jamais dans le code.
- **Phrase clé** : « Une nouvelle version du modèle ou de l'API se déploie de façon automatique, testée et réversible, en local comme dans le cloud. »

### 12:30 – 14:30 · Livrable 5 : le rapport, et la valeur clinique

**À l'écran** : `reports/rapport.pdf`, dans cet ordre : le tableau du résumé, la partie 5.4 (contrôles de sécurité et évaluation externe du triage), le tableau de la roadmap (partie 8).

**À dire, en deux temps** :

**Les signaux de valeur clinique** :

- 0,44 de bonnes réponses aux QCM à réponse unique, contre 0,21 pour une réponse naïve ;
- sur 60 cas de triage externes, une seule urgence vitale sous-triée sur 20 ;
- les cas bénins simples réorientés par une règle explicable, et chaque décision justifiée ;
- une réponse en moins d'une seconde, compatible avec l'accueil des urgences ;
- une traçabilité complète, qui permet l'audit de chaque réponse.

**Les conditions à réunir** :

- 4 recommandations dangereuses sur 12 cas pièges (par exemple, du potassium face à une hyperkaliémie, ou des antalgiques face à une torsion testiculaire probable) ;
- des maladies inventées jamais reconnues comme telles (hallucinations) ;
- un sur-triage fréquent : 23 bons niveaux sur 60 cas externes ; le modèle distingue mal une urgence vitale d'une urgence modérée ;
- pas encore de données de triage validées par des soignants.

**La roadmap** : un jeu de cas annotés par les infirmiers d'accueil du CHSA, puis un modèle de triage avec des règles validées par l'équipe médicale, puis une infrastructure HDS intégrée au SIH, et enfin un pilote « en parallèle » des soignants.

**Phrase clé** : « Le potentiel est là, mais la valeur clinique se démontre avec des données de triage et un pilote, pas avec un modèle plus grand. »

### 14:30 – 15:00 · Conclusion

Trois phrases :

1. La faisabilité technique est démontrée : la chaîne complète fonctionne, de bout en bout, de façon reproductible et traçable, en local comme dans le cloud.
2. La valeur clinique est prometteuse mais conditionnée : le modèle n'a pas appris le triage. Des règles déterministes le rendent cohérent sur les cas simples, mais il commet encore des erreurs qu'un outil clinique ne peut pas se permettre.
3. La prochaine étape : constituer avec les équipes du CHSA un jeu de triage annoté, puis un pilote sans effet sur la décision des soignants.

## 3. Préparation

**La veille** :

- Lancer `./scripts/demo.sh --no-pause` pour vérifier que tout répond.
- Enregistrer une courte vidéo ou des captures de chaque étape (plan de secours).
- Répéter avec un chronomètre : la partie sur le dataset (3 minutes) est celle qui risque le plus de déborder.

**Une heure avant** :

- Ouvrir un terminal Ubuntu : WSL, Docker et le runner doivent être actifs.
- Vérifier la pile locale : `./scripts/demo.sh 1` doit répondre `ok` pour l'API et vLLM.
- Fermer tout programme qui utilise le GPU (notebook, entraînement).
- Agrandir la police du terminal, pour qu'elle soit lisible à distance.

**Dix minutes avant, si la démo se fait dans le cloud** :

- `uv run python scripts/cloud.py start` : réveille l'Inference Endpoint (environ 3 minutes) et le Space.
- Après la présentation : `uv run python scripts/cloud.py stop`, pour arrêter la facturation du GPU.

**Onglets à ouvrir, dans l'ordre de passage** :

1. Le README sur GitHub
2. Les notebooks `prep_SFT_MediQAl_oeq.ipynb` et `prep_DPO_MediQAl.ipynb`
3. La page Hugging Face du dataset
4. MLflow sur DagsHub (runs SFT v4 et DPO v5, expérience `qwen3-1.7b-serving`)
5. La page Hugging Face du modèle
6. GitHub Actions (dernière exécution réussie)
7. Le PDF du rapport
8. En réserve : l'interface `scripts/ui.py`, pour tester un patient proposé par le jury

## 4. Questions probables du jury

| Question | Éléments de réponse |
|---|---|
| Pourquoi des QCM et pas du triage ? | Aucun des jeux proposés n'a de niveau de triage. Les cas retenus sont les plus proches du triage (patient décrit, constantes, urgence), avec des questions ouvertes pour apprendre à expliquer. Les QCM valident la chaîne et se mesurent automatiquement, grâce au corrigé. |
| Le DPO sert-il à quelque chose ? | Oui, de façon mesurable : préférences justes à 0,68 en français, propositions cochées en trop presque supprimées, sans régression. Mais il ne rend pas le modèle plus juste de façon significative. À terme, il faudrait un DPO sur les vraies erreurs du modèle, validées par des soignants. |
| Pourquoi une règle pour les motifs bénins, et pas le modèle ? | Le modèle distingue mal une urgence vitale d'une urgence modérée. Une correction statistique a été testée : elle sous-triait des urgences vitales sur des cas externes. Une règle fermée, explicable et testée (aucune urgence vitale déclassée sur 60 cas externes) est plus sûre, comme un protocole d'accueil. |
| Comment le triage a-t-il été évalué ? | Sur nos 11 cas de démonstration et 9 QCM de triage, puis sur 60 cas externes d'un jeu public, avec un niveau de référence fixé par son auteur (non validé médicalement) : une mesure indicative. |
| Pourquoi pas directement un modèle de 32B ? | Un modèle plus grand ne saurait toujours pas trier sans données de triage. Les données d'abord. |
| Comment garantir le RGPD en production ? | Anonymisation ciblée, journal d'audit sans texte patient, hébergement HDS ou sur site, pseudonymisation des dossiers historiques avec le délégué à la protection des données. |
| Pourquoi le modèle invente-t-il des réponses ? | Il est entraîné à toujours choisir une proposition. En production, il faudra une option d'abstention (« avis soignant requis ») et des garde-fous déterministes. |
| Quel coût pour passer à l'échelle ? | Environ 64 Go de mémoire GPU pour un modèle de 32B (deux GPU de 48 Go), moins s'il est quantifié. La chaîne CI/CD reste la même. La démo cloud actuelle coûte 0,50 $/h, seulement quand l'endpoint est allumé. |
| Les réponses sont-elles reproductibles ? | À 97–98 % : vLLM regroupe les requêtes, et 2 à 3 % des réponses peuvent varier. Le journal enregistre la réponse effectivement donnée. |
