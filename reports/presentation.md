# Présentation des livrables : plan et déroulé (15 minutes)

**Consigne** : faire la démonstration du POC, et démontrer la faisabilité technique et la valeur ajoutée clinique du système d'agent intelligent de triage.

**Principe** : pas de diapositives. On montre les cinq livrables du sujet directement à l'écran (terminal, dépôt, MLflow, Hugging Face, GitHub, rapport). On ouvre sur le service en marche, pour accrocher le jury, puis on remonte la chaîne pour montrer comment il a été construit.

**Outil** : le script `scripts/demo.sh` déroule toutes les démonstrations du terminal, étape par étape (Entrée pour passer à la suivante). Une étape seule se lance avec `./scripts/demo.sh <numéro>`.

## 1. Plan minuté

| Temps | Livrable | À l'écran | Étapes du script |
|---|---|---|---|
| 0:00 – 1:00 | Introduction | README du dépôt (haut de page) | — |
| 1:00 – 3:00 | 3. Endpoint : le résultat | Terminal | 1, 2, 3 |
| 3:00 – 6:00 | 1. Dataset bilingue anonymisé | README, notebooks SFT et DPO, fichier JSONL, rapport d'anonymisation | — |
| 6:00 – 8:30 | 2. Modèle SFT + DPO | MLflow (DagsHub), page Hugging Face | — |
| 8:30 – 11:00 | 3. Endpoint : la solidité | Terminal, MLflow (latence) | 4, 5, 6, 7 |
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
2. **Un patient décrit en langage naturel** (homme de 58 ans, douleur thoracique constrictive irradiant au bras gauche, sueurs) : « Urgence maximale », avec la justification, la conduite à tenir et les **questions à poser** (début de la douleur, irradiation, essoufflement), en moins d'une demi-seconde.
3. **Des constantes graves** (détresse respiratoire, SpO₂ 85 %, FR 34/min) : les **garde-fous** détectent les signes de gravité, qui imposeraient l'urgence maximale quel que soit l'avis du modèle.

**À dire** :

- Le soignant décrit le patient comme à l'accueil : motif, symptômes, constantes. L'API propose un niveau de priorité, l'explique, et dit quelles questions poser ensuite.
- Le modèle ne décide pas seul : il évalue le niveau sur un choix fermé, où il est fiable ; des règles sur les constantes peuvent seulement le relever.
- **Phrase clé** : « Un niveau de priorité expliqué en moins d'une seconde, avec les questions à poser : c'est le résultat. Voyons maintenant comment il a été construit. »

### 3:00 – 6:00 · Livrable 1 : le dataset bilingue anonymisé

**À l'écran** :

1. Le README, sections « Sources des données et licences » et « Schéma des métadonnées ».
2. Le notebook `prep_SFT_MediQAl_oeq.ipynb` : l'exemple d'une ligne (un cas et ses questions dans l'ordre), puis la cellule « Sélection pour le triage » (questions disponibles par score, cas retenus).
3. Le notebook `prep_DPO_MediQAl.ipynb` : la paire affichée (le cas, l'historique, la `chosen` et la `rejected`) et le tableau des types d'erreur.
4. Une ligne d'un fichier anonymisé :
   `head -1 data/v4/anonymized/dpo_mediqal_train.anonymized.jsonl | python3 -m json.tool`
5. Un rapport d'anonymisation :
   `python3 -m json.tool data/v4/anonymized/sft_mediqal_oeq_train.report.json`

**À dire** :

- **Sources** : MediQAl en français (cas cliniques rédigés par des enseignants : questions ouvertes et QCM avec corrigé) et UltraMedical en anglais (vignettes d'examen, questions de patients, test revu par des experts), sous licences ouvertes. MedQuAD et FrenchMedMCQA ont été écartés : aucun patient décrit.
- **Proche du triage** : chaque exemple décrit un patient, en priorité avec des constantes et des signes d'urgence. Sur près de 30 000 questions françaises, 2 998 sont retenues.
- **Une ligne par cas clinique** : le cas, puis ses questions dans l'ordre du récit, comme un entretien. Toutes les réponses sont apprises, chacune avec ce qui la précède.
- **Qualité plutôt que quantité** : chaque exemple est vérifié par le corrigé ou par la note. Par exemple, 111 541 paires anglaises d'origine pour 2 000 exemples SFT retenus.
- **Pas de fuite** : les splits officiels de MediQAl partagent des cas cliniques. La répartition est recalculée par cas ; aucun cas n'est vu à la fois en entraînement et en test, ni à la fois en SFT et en DPO.
- **DPO français** : la mauvaise réponse est construite à partir des distracteurs des enseignants, avec autant d'omissions que d'ajouts : l'erreur est plausible, le corrigé garantit qu'elle est fausse, et la longueur ne la trahit pas.
- **RGPD** : anonymisation Presidio ciblée sur les champs qui décrivent un patient. Un contrôle par sondage a montré que la moitié des remplacements des questions ouvertes étaient des termes médicaux (« <PERSON> coli ») : de nouvelles règles les écartent, 4 122 détections au total.
- **Phrase clé** : « 4 998 paires SFT et 3 629 paires DPO, orientées triage, vérifiées, sans fuite, anonymisées et documentées. »

### 6:00 – 8:30 · Livrable 2 : le modèle SFT + DPO

**À l'écran** :

1. MLflow sur DagsHub : les runs `sft-lora-r16-v3-seed42` et `dpo-lora-r16-v3-beta0.1-lr5e-05-seed42`, leurs paramètres et leurs courbes (`eval_loss`, `rewards/accuracies`). Le modèle servi est celui de la version 3 des données.
2. La page Hugging Face du modèle : la fiche modèle, les fichiers, la révision.

**À dire** :

- **SFT puis DPO, avec LoRA** : 1 % des poids entraînés, ce qui tient sur une carte de 10 Go. Le SFT apprend à répondre ; le DPO apprend à préférer la bonne réponse.
- **Démarche itérative** : petits runs d'abord, puis quatre versions des données. Chacune corrige un problème mesuré : propositions absentes du prompt, fuite de cas cliniques, biais de longueur, réponses à plusieurs lettres, puis, en v4, des données plus proches du triage. Le notebook d'entraînement est prêt pour la v4 (une ligne par cas, perte sur toutes les réponses) : c'est le prochain run.
- **Traçabilité** : chaque run garde ses hyperparamètres, sa graine et sa version de données. Les poids finaux sont publiés sur Hugging Face, à une révision fixée.
- **Phrase clé** : « Chaque modèle est reproductible : on sait sur quelles données et avec quels réglages il a été entraîné. »

### 8:30 – 11:00 · Livrable 3 : l'endpoint, la solidité

**À l'écran** : le terminal, étapes 4 à 7 du script, puis la latence dans MLflow (expérience `qwen3-1.7b-serving`).

4. **Une situation bénigne** (entorse de cheville, constantes normales) : « Urgence maximale » au lieu de « différée ». À annoncer avant de lancer : c'est une limite connue, le sur-triage.
5. **Sécurité** : sans clé, refus (401) ; avec de fausses balises `<|im_end|>`, refus (422) avant tout appel au modèle.
6. **Journal d'audit** : une ligne par requête, sans texte patient. La recherche du mot « constrictive » (tiré des symptômes) dans le journal donne 0 occurrence.
7. **Tests d'intégration en direct** : 15 tests réussis en quelques secondes, contre la pile réelle, dont les 5 urgences vitales jamais sous-triées.

**À dire** :

- Le sur-triage est l'erreur la moins dangereuse, mais elle engorgerait le service. C'est pour cela qu'un jeu de triage est la priorité de la roadmap.
- La faille des balises a été trouvée par nos tests de robustesse, puis corrigée : un utilisateur pouvait injecter un faux message système.
- **Latence** : 79 ms en médiane en local, 64 requêtes par seconde avec 50 utilisateurs simultanés, sur une seule carte graphique.
- **Phrase clé** : « Sécurisé, robuste, et tracé sans aucune donnée de santé. »

### 11:00 – 12:30 · Livrable 4 : le pipeline CI/CD

**À l'écran** :

1. GitHub Actions : une exécution réussie, avec ses trois jobs (`tests`, `image`, `deploy`).
2. Le job `deploy`, exécuté sur le runner `p14-gpu` (la machine GPU).
3. L'image de l'API sur GHCR, étiquetée par commit.

**À dire** :

- À chaque commit : tests unitaires, construction de l'image et publication. Sur `master` : déploiement sur la machine GPU, puis tests d'intégration contre la pile déployée.
- Le déploiement remplace l'API en quelques secondes, sans redémarrer vLLM. Revenir à une version précédente, c'est relancer le pipeline d'un ancien commit.
- Les secrets (clé d'API, jeton Hugging Face) sont dans GitHub, jamais dans le code.
- **Phrase clé** : « Une nouvelle version du modèle ou de l'API se déploie de façon automatique, testée et réversible. »

### 12:30 – 14:30 · Livrable 5 : le rapport, et la valeur clinique

**À l'écran** : `reports/rapport.pdf`, dans cet ordre : le tableau du résumé, la partie 5.4 (contrôles de sécurité), le tableau de la roadmap (partie 8).

**À dire, en deux temps** :

**Les signaux de valeur clinique** :

- 0,44 de bonnes réponses aux QCM à réponse unique, contre 0,24 pour une réponse naïve ;
- les 4 urgences vitales des cas de triage, toujours classées en urgence maximale : aucun sous-triage ;
- une réponse en moins d'une seconde, compatible avec l'accueil des urgences ;
- une traçabilité complète, qui permet l'audit de chaque réponse.

**Les conditions à réunir** :

- 3 recommandations dangereuses sur 12 cas pièges (par exemple, du potassium face à une hyperkaliémie) ;
- des maladies inventées jamais reconnues comme telles (hallucinations) ;
- un sur-triage des cas bénins ;
- pas encore de données de triage.

**La roadmap** : un jeu de cas annotés par les infirmiers d'accueil du CHSA, puis un modèle de triage avec garde-fous, puis une infrastructure HDS intégrée au SIH, et enfin un pilote « en parallèle » des soignants.

**Phrase clé** : « Le potentiel est là, mais la valeur clinique se démontre avec des données de triage et un pilote, pas avec un modèle plus grand. »

### 14:30 – 15:00 · Conclusion

Trois phrases :

1. La faisabilité technique est démontrée : la chaîne complète fonctionne, de bout en bout, de façon reproductible et traçable.
2. La valeur clinique est prometteuse mais conditionnée : le modèle n'a pas encore appris le triage, et il commet des erreurs qu'un outil clinique ne peut pas se permettre.
3. La prochaine étape : constituer avec les équipes du CHSA un jeu de triage annoté, puis un pilote sans effet sur la décision des soignants.

## 3. Préparation

**La veille** :

- Lancer `./scripts/demo.sh --no-pause` pour vérifier que tout répond.
- Enregistrer une courte vidéo ou des captures de chaque étape (plan de secours).
- Répéter avec un chronomètre : la partie sur le dataset (3 minutes) est celle qui risque le plus de déborder.

**Une heure avant** :

- Ouvrir un terminal Ubuntu : WSL, Docker et le runner doivent être actifs.
- Vérifier la pile : `./scripts/demo.sh 1` doit répondre `ok` pour l'API et vLLM.
- Fermer tout programme qui utilise le GPU (notebook, entraînement).
- Agrandir la police du terminal, pour qu'elle soit lisible à distance.

**Onglets à ouvrir, dans l'ordre de passage** :

1. Le README sur GitHub
2. Les notebooks `prep_SFT_MediQAl_oeq.ipynb` et `prep_DPO_MediQAl.ipynb`
3. MLflow sur DagsHub (runs SFT et DPO v3, expérience `qwen3-1.7b-serving`)
4. La page Hugging Face du modèle
5. GitHub Actions (dernière exécution réussie)
6. Le PDF du rapport

## 4. Questions probables du jury

| Question | Éléments de réponse |
|---|---|
| Pourquoi des QCM et pas du triage ? | Aucun jeu de triage ouvert. La v4 retient les cas les plus proches du triage (patient décrit, constantes, urgence) et ajoute des questions ouvertes pour apprendre à expliquer. Les QCM valident la chaîne et se mesurent automatiquement, grâce au corrigé. |
| Le DPO sert-il à quelque chose ? | En v3, il ne dégrade rien et corrige une erreur dangereuse (dose de paracétamol), mais son gain est surtout un raccourci (cocher moins). La v4 équilibre omissions et ajouts pour supprimer ce raccourci. À terme, il faudrait un DPO sur les vraies erreurs du modèle. |
| Pourquoi pas directement un modèle de 32B ? | Un modèle plus grand ne saurait toujours pas trier sans données de triage. Les données d'abord. |
| Comment garantir le RGPD en production ? | Anonymisation ciblée, journal d'audit sans texte patient, hébergement HDS ou sur site, pseudonymisation des dossiers historiques avec le délégué à la protection des données. |
| Pourquoi le modèle invente-t-il des réponses ? | Il est entraîné à toujours choisir une proposition. En production, il faudra une option d'abstention (« avis soignant requis ») et des garde-fous déterministes. |
| Quel coût pour passer à l'échelle ? | Environ 64 Go de mémoire GPU pour un modèle de 32B en bf16 (deux GPU de 48 Go), moins avec une quantification. La chaîne CI/CD reste la même. |
| Pourquoi le modèle servi n'est-il pas entraîné sur la v4 ? | La v4 corrige les données (triage, une ligne par cas, anonymisation) ; la pipeline d'entraînement est adaptée et vérifiée sans GPU, le run reste à lancer. |
| Les réponses sont-elles reproductibles ? | À 97–98 % : vLLM regroupe les requêtes, et 2 à 3 % des réponses peuvent varier. Le journal enregistre la réponse effectivement donnée. |
