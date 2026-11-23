---
license: cc-by-4.0
language: [fr, en]
task_categories: [text-generation]
tags: [medical, triage, sft, dpo]
size_categories: [1K<n<10K]
configs:
- config_name: sft_mediqal_oeq
  data_files:
  - {split: train, path: sft/sft_mediqal_oeq_train.jsonl}
  - {split: validation, path: sft/sft_mediqal_oeq_validation.jsonl}
  - {split: test, path: sft/sft_mediqal_oeq_test.jsonl}
- config_name: sft_mediqal_mcqu
  data_files:
  - {split: train, path: sft/sft_mediqal_mcqu_train.jsonl}
  - {split: validation, path: sft/sft_mediqal_mcqu_validation.jsonl}
  - {split: test, path: sft/sft_mediqal_mcqu_test.jsonl}
- config_name: sft_mediqal_mcqm
  data_files:
  - {split: train, path: sft/sft_mediqal_mcqm_train.jsonl}
  - {split: validation, path: sft/sft_mediqal_mcqm_validation.jsonl}
  - {split: test, path: sft/sft_mediqal_mcqm_test.jsonl}
- config_name: sft_ultramedical
  data_files:
  - {split: train, path: sft/sft_ultramedical_train.jsonl}
  - {split: validation, path: sft/sft_ultramedical_validation.jsonl}
  - {split: test, path: sft/sft_ultramedical_test.jsonl}
- config_name: dpo_mediqal
  data_files:
  - {split: train, path: dpo/dpo_mediqal_train.jsonl}
  - {split: validation, path: dpo/dpo_mediqal_validation.jsonl}
  - {split: test, path: dpo/dpo_mediqal_test.jsonl}
- config_name: dpo_ultramedical
  data_files:
  - {split: train, path: dpo/dpo_ultramedical_train.jsonl}
  - {split: validation, path: dpo/dpo_ultramedical_validation.jsonl}
  - {split: test, path: dpo/dpo_ultramedical_test.jsonl}
---

# Triage médical bilingue (fr / en) — SFT et DPO

Données d'entraînement du modèle [`0biyohan/qwen3-1.7b-medical-triage-fr`](https://huggingface.co/0biyohan/qwen3-1.7b-medical-triage-fr),
POC d'agent de triage aux urgences. Cas cliniques et questions médicales sélectionnés pour le triage (patient décrit,
constantes vitales, vocabulaire d'urgence), nettoyés et anonymisés.

| Config | Format | Langue | Contenu |
|---|---|---|---|
| `sft_mediqal_oeq` | `messages` | fr | 1 098 questions ouvertes (204 cas) |
| `sft_mediqal_mcqu` | `messages` | fr | 1 200 QCM à une réponse (293 cas) |
| `sft_mediqal_mcqm` | `messages` | fr | 700 QCM à plusieurs réponses (251 cas) |
| `sft_ultramedical` | `messages` | en | 2 000 questions |
| `dpo_mediqal` | `prompt` / `chosen` / `rejected` | fr | 2 156 paires : corrigé contre une erreur construite |
| `dpo_ultramedical` | `prompt` / `chosen` / `rejected` | en | 800 paires ; `test` : 756 paires revues par des experts |

```python
from datasets import load_dataset
ds = load_dataset("0biyohan/triage-medical-bilingual", "sft_mediqal_oeq", revision="v1.0")
```

- **Splits** : train / validation / test (70 / 10 / 20), tirés de l'empreinte du cas clinique : aucun cas commun
  entre splits, ni entre SFT et DPO.
- **Métadonnées** : `source`, `lang`, `case_id`, `pair_check` (mode de validation de la réponse), etc.
  [Schéma complet](https://github.com/ObiYohan/Projet_14#schéma-des-métadonnées).
- **Anonymisation (RGPD)** : noms de personnes remplacés par `<PERSON>` (Presidio + spaCy), sans table de
  correspondance ; comptes par fichier dans `anonymization/`.

## Sources et licences

| Source | Licence |
|---|---|
| [ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | CC-BY-4.0 |
| [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | MIT (agrège MedQA, MedMCQA, ChatDoctor : voir leurs conditions) |

Ce jeu dérivé est publié sous CC-BY-4.0, avec attribution des sources ci-dessus.

## Limites

Pas de relecture clinique humaine : la validation repose sur les corrigés, les notes GPT-4 et des contrôles
automatiques. Usage recherche uniquement ; ne pas utiliser pour un diagnostic ou un triage réel.
