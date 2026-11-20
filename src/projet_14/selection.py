"""Sélection des cas pour le SFT de triage (version 4) : critères communs aux corpus français et anglais.

Le corpus complet dépasse le besoin (environ 5 000 paires instruction-réponse). On garde les cas les plus proches du
triage, par **cas entier** (toutes ses questions), pour garder le récit cohérent et la répartition par cas :

1. **Patient décrit** (obligatoire) : un cas clinique (français) ou une vignette / un patient qui décrit ses symptômes
   (anglais). Le triage part toujours d'un patient.
2. **Score de pertinence** (0 à 2) : `constantes` (PA, FC, SpO2, température, Glasgow… dans le cas) + `urgence`
   (vocabulaire d'urgence ou de conduite à tenir dans le cas, les questions ou les réponses).
3. **Budget** de questions, réparti entre les splits (70 / 10 / 20) : dans chaque split, cas pris par score
   décroissant, puis dans un ordre pseudo-aléatoire fixe tiré de l'identifiant du cas.
4. **Plafond par groupe** (spécialité ou source) : une part maximale du budget, pour garder la diversité.

Critères volontairement simples et lisibles (expressions régulières) : chaque choix s'explique. Aucune relecture
clinique humaine n'est prévue dans le POC : la pertinence des cas retenus n'est pas validée par un soignant.
"""

import re
from collections import Counter

import pandas as pd

from projet_14.conversation import CASE_LABEL, QUESTION_LABEL
from projet_14.mediqal import SPLIT_SHARES

VITALS = {
    "fr": re.compile(
        r"\bPA\b|pression artérielle|tension artérielle|\d{2,3}\s*/\s*\d{2,3}\s*mm ?Hg|fréquence cardiaque|\bFC\b|pouls"
        r"|SpO2|saturation|fréquence respiratoire|température|°C|Glasgow", re.I),
    "en": re.compile(
        r"blood pressure|\bBP\b|\d{2,3}\s*/\s*\d{2,3}\s*mm ?Hg|pulse|heart rate|respiratory rate|respirations"
        r"|oxygen saturation|SpO2|temperature|°[CF]|\bGCS\b|Glasgow", re.I),
}
URGENCY = {
    "fr": re.compile(
        r"urgen|conduite à tenir|première(s)? mesure|en priorité|immédiat|hospitalis|réanimation|signes? de gravité"
        r"|gravité|SAMU|SMUR|choc|détresse|pronostic vital|orient", re.I),
    "en": re.compile(
        r"emergenc|urgent|immediate|next (best )?step|most appropriate (next|initial)|initial management|hospitali"
        r"|admi(t|ssion)|intensive care|\bICU\b|resuscitat|shock|life-threatening|unstable|red flag|triage"
        r"|ambulance|911", re.I),
}
# Anglais : vignette (« A 45-year-old man… ») ou patient qui parle de lui ou d'un proche
PATIENT_EN = re.compile(
    r"\b\d{1,3}[- ](year|month|week|day)s?[- ]old\b|\bI(?: am|'m) (?:a )?\d{1,3}\b"
    r"|\bmy (son|daughter|husband|wife|mother|father|mom|dad|baby|child|kid)\b|\bI (have|had|am having|get)\b", re.I)


def triage_flags(case_text: str, text: str, lang: str) -> tuple[bool, bool]:
    """(constantes dans le cas, vocabulaire d'urgence dans le texte complet : cas, questions, réponses)."""
    return bool(VITALS[lang].search(case_text)), bool(URGENCY[lang].search(text))


def case_table(examples: pd.DataFrame, lang: str, group: str) -> pd.DataFrame:
    """Une ligne par cas : questions, groupe (`group`), split, patient décrit, constantes, urgence, score.

    `examples` : lignes SFT (`case_id`, `split`, `n_questions`, `messages`, colonne `group`), un cas pouvant être
    coupé en plusieurs lignes.
    """
    rows = []
    for cid, part in examples.groupby("case_id", sort=False):
        first_user = part["messages"].iloc[0][1]["content"]  # [0] : système ; [1] : cas (ou prompt) et 1re question
        text = "\n".join(m["content"] for messages in part["messages"] for m in messages[1:])
        if lang == "fr":
            patient = first_user.startswith(CASE_LABEL)
            case_text = first_user.split(f"\n\n{QUESTION_LABEL}", 1)[0] if patient else ""
        else:
            patient = bool(PATIENT_EN.search(first_user))
            case_text = first_user
        vitals, urgency = triage_flags(case_text, text, lang)
        rows.append({"case_id": cid, "split": part["split"].iloc[0], group: part[group].iloc[0],
                     "n_questions": int(part["n_questions"].sum()), "patient": patient,
                     "constantes": vitals, "urgence": urgency, "score": int(vitals) + int(urgency)})
    return pd.DataFrame(rows)


def select_cases(cases: pd.DataFrame, budget: int, max_share: float, group: str, min_score: int = 1) -> pd.DataFrame:
    """Cas retenus : patient décrit, score ≥ `min_score`, au plus `budget` questions et `max_share` du budget par groupe.

    Le budget est réparti entre les splits (70 / 10 / 20, `SPLIT_SHARES`) : la sélection garde leurs proportions.
    """
    pool = cases[cases["patient"] & (cases["score"] >= min_score)]
    # Ordre fixe et pseudo-aléatoire : 1er tirage de l'identifiant (le split vient du 2e : indépendant)
    pool = pool.assign(order=pool["case_id"].map(lambda cid: int(cid[:6], 16)))
    pool = pool.sort_values(["score", "order"], ascending=[False, True])
    keep = []
    for split, share in SPLIT_SHARES.items():
        split_budget = round(budget * share)
        cap = max_share * split_budget
        total, per_group = 0, Counter()
        for r in pool[pool["split"] == split].itertuples():
            n, g = r.n_questions, getattr(r, group)
            if total + n > split_budget or per_group[g] + n > cap:
                continue
            keep.append(r.case_id)
            total += n
            per_group[g] += n
            if total >= split_budget:
                break
    return cases[cases["case_id"].isin(keep)]

