"""Mise en forme commune des exemples SFT (version 4) : une conversation par cas clinique.

Un cas clinique de MediQAl porte plusieurs questions qui forment un récit (examens, résultats, évolution…). Chaque
cas devient une conversation, dans l'ordre des questions :

    système   : consigne générale
    user      : « Cas clinique : … » + « Question : … » (+ propositions et consigne pour un QCM)
    assistant : réponse
    user      : « Question : … »   (le cas n'est pas répété)
    assistant : réponse
    …

Une ligne du JSONL = une conversation (ou une tranche, si elle dépasse `MAX_CHARS`). À l'entraînement, la perte porte
sur toutes les réponses de l'assistant (`assistant_only_loss` de TRL) : chaque réponse est apprise avec l'historique
qui la précède, sans dupliquer le texte. Chaque tranche reprend le cas clinique en tête de son premier message.

Le split (train / validation / test) est tiré de l'identifiant du cas (`projet_14.mediqal`) : identique dans tous
les notebooks, aucun cas partagé entre entraînement et évaluation.
"""

from projet_14.mediqal import MULTIPLE_ANSWERS, OPTIONS_HEADER, SINGLE_ANSWER, case_assignment, case_id

SYSTEM_PROMPT = "Tu es un assistant médical expert. Réponds de façon précise et concise."
SYSTEM_PROMPT_EN = "You are an expert medical assistant. Answer accurately and clearly."  # corpus anglais (UltraMedical)
CASE_LABEL = "Cas clinique :"
QUESTION_LABEL = "Question :"
MAX_CHARS = 3_500  # ~1 000 tokens : limite d'un exemple d'entraînement


def split_of(text: str) -> str:
    """Split d'un cas clinique (ou d'une question sans cas) : train, validation ou test (70 / 10 / 20)."""
    return case_assignment(case_id(text))[1]


def user_message(question: str, case: str | None = None, options: dict | None = None,
                 multiple: bool | None = None) -> str:
    """Message user : cas clinique (premier message seulement), question, puis propositions et consigne (QCM)."""
    parts = [f"{CASE_LABEL} {case}"] if case else []
    parts.append(f"{QUESTION_LABEL} {question}")
    if options:
        parts.append(OPTIONS_HEADER + "\n" + "\n".join(f"{letter}. {text}" for letter, text in options.items()))
        parts.append(MULTIPLE_ANSWERS if multiple else SINGLE_ANSWER)
    return "\n\n".join(parts)


def conversation_windows(case: str | None, turns: list[dict]) -> list[tuple[int, list[dict]]]:
    """Une conversation par cas (une ligne du JSONL), coupée en tranches consécutives si elle dépasse `MAX_CHARS`.

    `turns` : questions dans l'ordre, avec `question` et `answer` (et `options`, `multiple` pour un QCM).
    Renvoie `(indice de la première question, messages)` pour chaque tranche. Chaque question figure dans une seule
    tranche, et chaque tranche reprend le cas clinique en tête de son premier message. À l'entraînement, toutes les
    réponses de l'assistant sont apprises (`assistant_only_loss`), chacune avec l'historique qui la précède.
    """
    windows, start = [], 0
    while start < len(turns):
        end = start + 1  # au moins une question par tranche, même trop longue (écartée ensuite par le notebook)
        while end < len(turns) and _length(case, turns[start:end + 1]) <= MAX_CHARS:
            end += 1
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for j, turn in enumerate(turns[start:end]):
            messages.append({"role": "user", "content": user_message(
                turn["question"], case if j == 0 else None, turn.get("options"), turn.get("multiple"))})
            messages.append({"role": "assistant", "content": turn["answer"]})
        windows.append((start, messages))
        start = end
    return windows


def _length(case: str | None, turns: list[dict]) -> int:
    # Longueur des messages tels qu'ils seront écrits (cas en tête du premier message gardé)
    return sum(
        len(user_message(t["question"], case if j == 0 else None, t.get("options"), t.get("multiple"))) + len(t["answer"])
        for j, t in enumerate(turns)
    )
