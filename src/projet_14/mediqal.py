"""Outils communs aux notebooks MediQAl (SFT mcqu, SFT mcqm, DPO mcqm) et à l'évaluation du modèle.

- **Prompt** : cas clinique, question, puis les propositions A à E après l'en-tête `OPTIONS_HEADER`, et une
  consigne sur le nombre de bonnes réponses (une seule en `mcqu`, plusieurs en `mcqm`).
- **Réponse** : lettre(s) et texte de la ou des bonnes propositions (« C. Pneumonie lobaire aiguë »).
  `answer_letters` relit les lettres d'une réponse, générée ou attendue : l'exactitude se mesure sur les lettres.
- **Répartition par cas clinique** : MediQAl n'a pas d'identifiant de cas, et ses splits officiels partagent des
  cas (un même cas sert à plusieurs questions, parfois dans le train et dans le test). `case_id` calcule un
  identifiant à partir du texte du cas ; `case_assignment` en tire, toujours de la même façon, un usage (SFT ou
  DPO) et un split (train / validation / test). Un cas, avec toutes ses questions `mcqu` et `mcqm`, n'est donc
  jamais partagé entre SFT et DPO, ni entre entraînement et évaluation.

Usage depuis un notebook :
    from projet_14.mediqal import assign_cases, user_content, format_answer
"""

import hashlib
import re

import pandas as pd

LETTERS = "ABCDE"
OPTION_COLUMNS = [f"answer_{letter.lower()}" for letter in LETTERS]
SYSTEM_PROMPT = "Tu es un assistant médical expert. Réponds de façon précise et concise."

# En-tête de la liste des propositions dans le prompt. L'anonymisation s'arrête à cet en-tête
# (rgpd.run(..., keep_after=OPTIONS_HEADER)) : les propositions sont des énoncés médicaux sans donnée patient.
OPTIONS_HEADER = "Propositions :"

# Consigne en fin de prompt : nombre de bonnes réponses attendues (mcqu : toujours 1 ; mcqm : toujours 2 à 5)
SINGLE_ANSWER = "Une seule proposition est exacte."
MULTIPLE_ANSWERS = "Plusieurs propositions sont exactes (au moins deux)."

# Part des cas cliniques réservés au DPO (le reste va au SFT), puis découpage de chaque usage
DPO_SHARE = 0.5
SPLIT_SHARES = {"train": 0.7, "validation": 0.1, "test": 0.2}

# Ligne de réponse qui commence par une lettre : « C. … », « - A. … », « B) … »
ANSWER_LINE = re.compile(r"^\s*(?:[-*•]\s*)?([A-E])\s*[.)]", re.MULTILINE)


def clean_text(text: str) -> str:
    """Retire les espaces en début / fin de ligne et les lignes vides en trop."""
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def case_id(clinical_case: str) -> str:
    """Identifiant d'un cas clinique : hash de son texte (espaces et casse ignorés)."""
    normalized = re.sub(r"\s+", " ", clinical_case).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def case_assignment(cid: str) -> tuple[str, str]:
    """Usage (« sft » ou « dpo ») et split d'un cas, tirés de son identifiant : identiques d'un notebook à l'autre."""
    usage_draw = int(cid[:6], 16) / 16**6  # deux tirages indépendants dans [0, 1)
    split_draw = int(cid[6:12], 16) / 16**6
    usage = "dpo" if usage_draw < DPO_SHARE else "sft"
    cumulative = 0.0
    for split, share in SPLIT_SHARES.items():
        cumulative += share
        if split_draw < cumulative:
            break
    return usage, split


def assign_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `case_id`, `usage` et `split` aux questions avec cas clinique (vides pour les autres)."""
    case_ids = df["clinical_case"].map(case_id, na_action="ignore")
    assignment = case_ids.map(case_assignment, na_action="ignore")
    return df.assign(case_id=case_ids, usage=assignment.str[0], split=assignment.str[1])


def options_text(row) -> str:
    """Propositions A à E, une par ligne (les propositions vides sont omises)."""
    return "\n".join(
        f"{letter}. {row[column]}" for letter, column in zip(LETTERS, OPTION_COLUMNS) if row[column].strip()
    )


def user_content(row, *, multiple_answers: bool) -> str:
    """Message user : cas clinique (s'il y en a un), question, propositions, puis le nombre de réponses attendues.

    Sans cette consigne, `mcqu` et `mcqm` ont le même format de prompt : au run v2, le modèle donnait plusieurs
    lettres à 25 % des questions à réponse unique. La consigne vient après l'en-tête des propositions : elle n'est
    pas anonymisée (texte fixe).
    """
    context = row["question"] if pd.isna(row["clinical_case"]) else f"{row['clinical_case']}\n\n{row['question']}"
    instruction = MULTIPLE_ANSWERS if multiple_answers else SINGLE_ANSWER
    return f"{context}\n\n{OPTIONS_HEADER}\n{options_text(row)}\n\n{instruction}"


def parse_user_content(text: str) -> dict:
    """Inverse de `user_content` : cas clinique, question, propositions et type de QCM d'un message user.

    Sert à rejouer les QCM du jeu de test contre l'API (tests d'intégration, mesures de latence). La question est
    le dernier paragraphe avant les propositions ; ce qui précède est le cas clinique.
    """
    context, rest = text.split(f"\n\n{OPTIONS_HEADER}\n")
    options, instruction = rest.rsplit("\n\n", 1)
    *case, question = context.split("\n\n")
    # « A. texte » en début de ligne ; une proposition peut s'étendre sur plusieurs lignes
    parts = re.split(rf"^([{LETTERS}])\. ", options, flags=re.MULTILINE)[1:]
    return {
        "cas_clinique": "\n\n".join(case) or None,
        "question": question,
        "propositions": {letter: option.rstrip("\n") for letter, option in zip(parts[::2], parts[1::2])},
        "reponses_multiples": instruction == MULTIPLE_ANSWERS,
    }


def correct_letters(row) -> list[str]:
    """Lettres des bonnes réponses d'après le corrigé, dans l'ordre A à E."""
    return sorted(letter.strip().upper() for letter in row["correct_answers"].split(","))


def format_answer(row, letters, as_list: bool) -> str:
    """Réponse « lettre. texte » : une ligne (réponse unique) ou une liste à puces (réponses multiples)."""
    lines = [f"{letter}. {row[f'answer_{letter.lower()}']}" for letter in sorted(letters)]
    return "\n".join(f"- {line}" for line in lines) if as_list else "\n".join(lines)


def answer_letters(text: str) -> frozenset[str]:
    """Lettres données dans une réponse (une par ligne commençant par une lettre)."""
    return frozenset(ANSWER_LINE.findall(text))
