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
