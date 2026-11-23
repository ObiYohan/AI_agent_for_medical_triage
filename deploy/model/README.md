---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Base
language:
  - fr
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - triage
  - qwen3
  - lora
  - sft
  - dpo
  - trl
  - vllm
datasets:
  - 0biyohan/triage-medical-bilingual
  - ANR-MALADES/MediQAl
  - TsinghuaC3I/UltraMedical-Preference
---

# Qwen3-1.7B triage médical (français) : SFT LoRA + DPO

Modèle d'un **POC d'agent IA de triage** pour le service des urgences d'un hôpital public (projet de formation).
`Qwen/Qwen3-1.7B-Base` est adapté au domaine médical par fine-tuning supervisé (SFT) avec LoRA, puis aligné par
préférences (DPO), sur des cas cliniques et des questions sélectionnés pour leur proximité avec le triage. Les deux
adaptateurs sont fusionnés dans ces poids (bf16, 3,4 Go).

Dans le POC, le modèle propose un **niveau de priorité sur un choix fermé** (urgence maximale / modérée / différée),
lu dans les probabilités de sa réponse. Des règles déterministes l'encadrent : des constantes vitales hors seuils
imposent l'urgence maximale, une liste fermée de motifs bénins donne l'urgence différée.

> **Démonstration technique uniquement.** Ce modèle ne pose pas de diagnostic, ne remplace pas un avis médical et
> n'est pas validé cliniquement. Il n'a vu aucun exemple étiqueté par niveau de triage.

## Utilisation

Gabarit de conversation fourni (`chat_template.jinja`, ChatML). La réponse se termine par `<|endoftext|>`, pas par
`<|im_end|>` (jamais entraîné dans Qwen3-Base ; LoRA ne modifie pas les embeddings).

Message système : `Tu es un assistant médical expert. Réponds de façon précise et concise.`

**Triage** (format QCM de l'entraînement) : la lettre générée donne le niveau ; avec `max_tokens: 1` et
`logprobs`, les probabilités de A, B et C donnent la confiance.

```
<cas clinique : âge, motif, symptômes, antécédents, constantes>

Quel niveau de priorité attribuez-vous à ce patient aux urgences ?

Propositions :
A. urgence maximale
B. urgence modérée
C. urgence différée

Une seule proposition est exacte.
```

**QCM médicaux** : même format, propositions A à E, consigne « Une seule proposition est exacte. » ou « Plusieurs
propositions sont exactes (au moins deux). ». Réponse : `C. texte de la proposition` ou une liste `- A. …`.

**Avec vLLM** (jeton Hugging Face en lecture, le dépôt étant privé) :

```bash
vllm serve 0biyohan/qwen3-1.7b-medical-triage-fr --revision 4af91a7aa052c092cfb64568d441b84f8f4a88e8 \
  --served-model-name qwen3-medical --max-model-len 2048
```

Puis `POST /v1/chat/completions` avec `temperature: 0`.

## Entraînement

Données : [`0biyohan/triage-medical-bilingual`](https://huggingface.co/datasets/0biyohan/triage-medical-bilingual)
(SFT v4, DPO v5), anonymisées avec Presidio et spaCy, réparties par cas clinique (aucun cas partagé entre SFT et
DPO, ni entre entraînement et évaluation).

| Étape | Données (split train) | Réglages |
|---|---|---|
| **SFT** (LoRA) | 3 500 questions : cas cliniques MediQAl (questions ouvertes, QCM à une ou plusieurs réponses), un cas par ligne avec toutes ses questions, et 1 400 questions UltraMedical en anglais | r = 16, alpha = 32, dropout 0,05, toutes les couches linéaires ; perte sur toutes les réponses ; 1 époque, lr 2e-4 (cosinus), batch effectif 16, 1 536 tokens ; seed 42 |
| **DPO** (LoRA) | 2 234 paires : 1 507 QCM MediQAl (`rejected` = corrigé avec une substitution, une omission ou un ajout) et 727 paires UltraMedical-Preference vérifiées par le corrigé | perte DPO (beta 0,1) + vraisemblance de la `chosen` ; lr 5e-5, 1 époque, batch effectif 16 ; référence = modèle SFT |

Le terme de vraisemblance maintient le format : sans lui, une réponse sur dix ajoutait une explication.
Outils : PyTorch, Transformers, PEFT, TRL, MLflow.

## Résultats

**Triage**, sur 60 cas publics ([clinical-triage-bilingual](https://huggingface.co/datasets/BenoitJT-GIRARD/clinical-triage-bilingual),
20 par niveau, niveaux fixés par l'auteur du jeu, sans validation médicale), modèle seul :

| Bons niveaux | Urgences vitales sous-triées | Cas sur-triés |
|---|---|---|
| 23 / 60 | 1 / 20 | 36 / 60 |

Le modèle **sur-trie** (a priori de 0,74 pour l'urgence maximale sur un cas vide) et distingue mal une urgence vitale
d'une urgence modérée. Une calibration de cet a priori gagne 5 niveaux mais sous-trie plus de la moitié des urgences
vitales : elle n'est pas utilisée.

**QCM médicaux**, 828 QCM français du test (cas jamais vus), réponse exactement égale au corrigé :

| Jeu | Questions | SFT seul | SFT + DPO | Réponse la plus fréquente |
|---|---|---|---|---|
| MediQAl `mcqu` (une réponse) | 389 | 0,432 | **0,445** | 0,211 |
| MediQAl `mcqm` (2 à 5 réponses) | 439 | 0,109 | 0,121 | 0,059 |

- Pas de régression due au DPO (McNemar, p = 0,36) ; format respecté à 99 % ; propositions cochées en trop :
  0,62 → 0,02 par question.
- Préférences DPO sur les paires de test : 0,68 en français, 0,74 en anglais (test revu par des experts).

**Latence** (vLLM, une requête de triage, médiane) : 82 ms en local, 0,55 s sur GPU T4 dans le cloud, réseau compris.

## Limites

- **Triage non appris directement** : pas de données étiquetées par niveau ; sur-triage fréquent, qui engorgerait
  le service ; niveaux à valider par des infirmiers organisateurs de l'accueil avant tout usage.
- **Connaissances limitées** d'un modèle de 1,7 milliard de paramètres : moins d'une réponse juste sur deux aux QCM
  à réponse unique.
- **Format fermé uniquement** : en question libre, les réponses en français ne sont pas fiables.
- Un seul entraînement (seed 42) : des écarts de quelques points entre versions ne sont pas significatifs.
- Version précédente (données v3) : révision `476fed3104b92d446e5c915a6adea7338825c255`.

## Licences

- Modèle de base : [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), Apache 2.0.
- Données : [ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) (CC-BY-4.0) et
  [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) (MIT ;
  agrège MedQA, MedMCQA et ChatDoctor, soumis à leurs propres conditions).
