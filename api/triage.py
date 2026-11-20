"""Triage d'un patient décrit en langage naturel : niveau de priorité, explication et questions complémentaires.

Le modèle n'a appris le français que sur des questions à choix multiples : en question libre, il produit des
réponses incohérentes. Il est donc utilisé là où il est fiable, sur un choix fermé (maximale / modérée / différée),
à partir d'un cas clinique rédigé à partir des informations reçues. Le reste est déterministe :

- calibration (expérimentale, désactivée par défaut) : le modèle surestime l'urgence ; ses probabilités peuvent
  être corrigées par celles d'un cas vide. Sur 60 cas externes (scripts/evaluate_triage.py), elle réduit le
  sur-triage mais sous-trie plus de la moitié des urgences vitales : le modèle distingue mal une urgence vitale
  d'une urgence modérée ;
- garde-fous : des constantes vitales hors seuils imposent l'urgence maximale. Ils peuvent seulement relever le
  niveau proposé par le modèle, jamais l'abaisser ;
- motifs bénins : à l'inverse, une liste fermée de motifs (rhume, mal de gorge, petite plaie superficielle…) chez
  un adulte, avec des constantes complètes et normales et sans aucun signe d'alerte dans le texte, donne une urgence
  différée (réorientation vers la médecine de ville), comme le ferait un protocole d'accueil par motif ;
- questionnaire adaptatif : questions complémentaires selon le motif, et constantes manquantes à mesurer ;
- conduite à tenir : texte fixe pour chaque niveau.

Les seuils et les questions sont des valeurs de démonstration, inspirées des critères de gravité usuels : ils sont
à définir et à valider par l'équipe médicale avant tout usage clinique.
"""

import math
import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

LEVELS = {"A": "urgence maximale", "B": "urgence modérée", "C": "urgence différée"}
SEVERITY = {"urgence maximale": 0, "urgence modérée": 1, "urgence différée": 2}  # 0 = le plus grave
QUESTION = "Quel niveau de priorité attribuez-vous à ce patient aux urgences ?"

# Calibration (expérimentale, TRIAGE_CALIBRATION=1) : le modèle n'a jamais appris le triage, et il a un a priori fort
# pour l'urgence maximale (0,74 pour un patient sans aucune information). Ses probabilités sont divisées par celles de
# ce cas vide (calibration contextuelle). Désactivée par défaut : elle sous-trie des urgences vitales.
NEUTRAL_CASE = "Un patient consulte aux urgences."
MIN_LETTER_MASS = 0.5  # probabilité minimale des lettres A, B, C au premier token : en dessous, réponse inexploitable

CONDUITE = {
    "urgence maximale": "Installation immédiate en salle d'urgence vitale et appel du médecin.",
    "urgence modérée": "Évaluation médicale rapide, dans l'heure ; surveillance des constantes en attendant.",
    "urgence différée": "Prise en charge non urgente : réorientation possible vers le médecin traitant ou une maison "
                        "médicale de garde ; réévaluation immédiate en cas d'aggravation.",
}

# Garde-fous : (constante, condition, signe de gravité affiché)
RED_FLAGS = [
    ("spo2", lambda v: v < 90, "saturation en oxygène à {v} % (< 90 %)"),
    ("pas", lambda v: v < 90, "pression artérielle systolique à {v} mmHg (< 90)"),
    ("fc", lambda v: v > 130, "fréquence cardiaque à {v}/min (> 130)"),
    ("fc", lambda v: v < 40, "fréquence cardiaque à {v}/min (< 40)"),
    ("fr", lambda v: v > 30, "fréquence respiratoire à {v}/min (> 30)"),
    ("fr", lambda v: v < 8, "fréquence respiratoire à {v}/min (< 8)"),
    ("temperature", lambda v: v >= 40, "température à {v} °C (≥ 40)"),
    ("temperature", lambda v: v < 35, "température à {v} °C (< 35)"),
    ("conscience", lambda v: v == "alteree", "conscience altérée"),
]

# Motifs bénins (protocole d'accueil) : (motif affiché, mots-clés du motif ou des symptômes)
BENIGN_MOTIFS = [
    ("rhume ou rhinopharyngite", ("rhume", "enrhum", "rhinopharyng", "nez qui coule", "nez bouché", "éternu")),
    ("mal de gorge", ("mal de gorge", "maux de gorge", "gorge irritée", "angine", "pharyngite")),
    ("petite plaie superficielle", ("petite plaie", "plaie superficielle", "égratignure", "écorchure", "éraflure")),
    ("piqûre de moustique", ("piqûre de moustique", "piqué par un moustique", "piquée par un moustique")),
    ("renouvellement d'ordonnance", ("renouvellement", "renouveler", "ordonnance")),
]
# Signes d'alerte : un seul suffit à écarter la règle. Recherchés dans tout le texte (motif, symptômes, antécédents),
# après retrait des passages niés (« pas de fièvre », « sans essoufflement », « ne saigne plus »)
ALARM_WORDS = (
    "essouffl", "dyspn", "respir", "étouff", "thorac", "poitrine", "malaise", "connaissance", "inconscien", "confus",
    "somnol", "convuls", "raideur", "nuque", "purpura", "tache", "saign", "sang", "gonfl", "œdème", "oedème",
    "avaler", "voix", "paralys", "faiblesse", "parler", "mots", "vomi", "intense", "violent", "brutal", "pire",
    "fièvre élevée", "frisson", "suicid", "en finir", "mourir", "grossesse", "enceinte", "anticoag",
    "immunodépr", "chimio", "greff", "cancer", "diabét", "insuffisan", "profond", "morsure", "rouille",
)
# Portée d'une négation : jusqu'à la ponctuation ou à une conjonction (« pas de fièvre mais douleur thoracique » :
# la douleur thoracique reste un signe d'alerte)
_SPAN = r"(?:(?!\b(?:mais|avec|et|puis|or|cependant|pourtant|sauf)\b)[^,.;:!?])*"
NEGATION = re.compile(rf"\b(?:pas\s+d['’e]|sans|ni)\s*{_SPAN}|\bne\s+\w+\s+(?:pas|plus|jamais)\b", re.IGNORECASE)
BENIGN_AGE = (15, 75)  # adultes seulement : enfants et personnes âgées relèvent d'autres seuils
# Constantes exigées, et leurs bornes normales (valeurs de démonstration, à valider par l'équipe médicale)
BENIGN_VITALS = {"fc": (50, 100), "pas": (100, 160), "spo2": (95, 100), "temperature": (36.0, 38.4)}
BENIGN_OPTIONAL = {"fr": (10, 22), "eva": (0, 3)}  # bornes vérifiées si la valeur est donnée

# Questionnaire adaptatif : questions selon le motif (mots-clés), puis constantes manquantes
QUESTIONS_PAR_MOTIF = [
    (("thorac", "poitrine"), ["Depuis quand la douleur a-t-elle commencé ?",
                              "La douleur irradie-t-elle (bras, mâchoire, dos) ?",
                              "Y a-t-il un essoufflement ou des sueurs ?"]),
    (("respir", "dyspn", "essouffl"), ["L'essoufflement est-il apparu brutalement ?",
                                       "Le patient peut-il parler en phrases complètes ?",
                                       "Y a-t-il de la fièvre ou une douleur thoracique ?"]),
    (("abdo", "ventre"), ["Où la douleur est-elle localisée ?",
                          "Y a-t-il de la fièvre, des vomissements ou du sang dans les selles ?",
                          "Une grossesse est-elle possible ?"]),
    (("tête", "céphal", "migraine"), ["Le mal de tête est-il apparu brutalement, en coup de tonnerre ?",
                                      "Y a-t-il une raideur de la nuque, une fièvre ou un trouble neurologique ?"]),
    (("chute", "trauma", "fracture", "entorse", "plaie", "accident"),
     ["Y a-t-il eu un choc à la tête ou une perte de connaissance ?",
      "Le patient prend-il un anticoagulant ?",
      "Peut-il bouger le membre et prendre appui ?"]),
    (("fièvre", "fievre"), ["Depuis combien de jours dure la fièvre ?",
                            "Y a-t-il des taches sur la peau qui ne s'effacent pas à la pression ?",
                            "Le patient est-il immunodéprimé ?"]),
]
QUESTIONS_GENERALES = ["Depuis quand les symptômes ont-ils commencé ?",
                       "Le patient a-t-il des antécédents ou des traitements en cours ?"]
CONSTANTES_A_MESURER = {
    "fc": "Mesurer la fréquence cardiaque.",
    "pas": "Mesurer la pression artérielle.",
    "spo2": "Mesurer la saturation en oxygène.",
    "temperature": "Prendre la température.",
}


SPECIAL_TOKEN = re.compile(r"<\|[^|<>]*\|>")


def _no_special_tokens(text: str) -> str:
    # Les balises du gabarit (<|im_start|>, <|im_end|>, <|endoftext|>…) écrites dans le texte deviendraient de vrais
    # tokens spéciaux : l'utilisateur pourrait fermer son message et injecter un faux message système
    if SPECIAL_TOKEN.search(text):
        raise ValueError("balise de gabarit interdite (<|…|>)")
    return text


Safe = AfterValidator(_no_special_tokens)


class Constantes(BaseModel):
    fc: int | None = Field(default=None, ge=20, le=250, description="fréquence cardiaque (/min)")
    pas: int | None = Field(default=None, ge=40, le=300, description="pression artérielle systolique (mmHg)")
    pad: int | None = Field(default=None, ge=20, le=200, description="pression artérielle diastolique (mmHg)")
    spo2: int | None = Field(default=None, ge=50, le=100, description="saturation en oxygène (%)")
    fr: int | None = Field(default=None, ge=4, le=80, description="fréquence respiratoire (/min)")
    temperature: float | None = Field(default=None, ge=30, le=45, description="température (°C)")
    eva: int | None = Field(default=None, ge=0, le=10, description="douleur sur 10")
    conscience: Literal["normale", "alteree"] | None = None


class TriageRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    sexe: Literal["F", "M"] | None = None
    motif: Annotated[str, Safe] = Field(min_length=1, max_length=300, description="motif de recours")
    symptomes: Annotated[str, Safe] = Field(min_length=1, max_length=3_000, description="description libre")
    antecedents: Annotated[str, Safe] | None = Field(default=None, max_length=1_000)
    constantes: Constantes = Constantes()


class TriageResponse(BaseModel):
    request_id: str
    niveau: str
    source: str
    niveau_modele: str | None
    niveau_modele_brut: str | None = None  # avant calibration
    signes_de_gravite: list[str]
    justification: str
    conduite: str
    questions_complementaires: list[str]
    modele: str
    latence_ms: int
    avertissement: str


def red_flags(constantes: Constantes) -> list[str]:
    flags = []
    for field, condition, label in RED_FLAGS:
        value = getattr(constantes, field)
        if value is not None and condition(value):
            flags.append(label.format(v=value))
    return flags


def clinical_case(request: TriageRequest) -> str:
    # Cas clinique rédigé à partir des informations reçues, dans le style des cas MediQAl de l'entraînement
    who = {"F": "Une femme", "M": "Un homme"}.get(request.sexe, "Un patient")
    parts = [f"{who} de {request.age} ans consulte aux urgences pour : {request.motif.strip().rstrip('.')}."]
    if request.antecedents:
        parts.append(f"Antécédents : {request.antecedents.strip().rstrip('.')}.")
    parts.append(f"Symptômes : {request.symptomes.strip().rstrip('.')}.")
    c = request.constantes
    measures = [
        f"fréquence cardiaque {c.fc}/min" if c.fc is not None else None,
        (f"pression artérielle {c.pas}/{c.pad} mmHg" if c.pad is not None else f"pression artérielle systolique {c.pas} mmHg")
        if c.pas is not None else None,
        f"saturation {c.spo2} %" if c.spo2 is not None else None,
        f"fréquence respiratoire {c.fr}/min" if c.fr is not None else None,
        f"température {c.temperature} °C" if c.temperature is not None else None,
        f"douleur {c.eva}/10" if c.eva is not None else None,
        "conscience altérée" if c.conscience == "alteree" else ("conscience normale" if c.conscience else None),
    ]
    measures = [m for m in measures if m]
    if measures:
        parts.append("Constantes : " + ", ".join(measures) + ".")
    return " ".join(parts)


def prompt_row(request: TriageRequest | None) -> dict:
    # Format des données d'entraînement (cas, question, propositions) : le niveau est un choix fermé à trois options.
    # Sans patient : le cas vide de la calibration
    row = {"clinical_case": clinical_case(request) if request else NEUTRAL_CASE, "question": QUESTION}
    for letter in "ABCDE":
        row[f"answer_{letter.lower()}"] = LEVELS[letter].capitalize() if letter in LEVELS else ""
    return row


def level_probabilities(top_logprobs: list[dict]) -> dict[str, float] | None:
    # Probabilités des trois niveaux d'après le premier token généré (lettre A, B ou C), normalisées ; None si le
    # modèle ne répond pas par une lettre
    mass = dict.fromkeys(LEVELS, 0.0)
    for candidate in top_logprobs:
        if candidate["token"] in mass:
            mass[candidate["token"]] += math.exp(candidate["logprob"])
    total = sum(mass.values())
    if total < MIN_LETTER_MASS:
        return None
    return {LEVELS[letter]: p / total for letter, p in mass.items()}


def model_levels(probabilities: dict[str, float], prior: dict[str, float] | None) -> tuple[str, str]:
    """Niveau brut (le plus probable) et niveau calibré (le plus probable une fois l'a priori retiré).

    La calibration abaisse le niveau d'un cran au plus : une urgence maximale brute ne devient jamais différée.
    """
    raw = max(probabilities, key=probabilities.get)
    if not prior:
        return raw, raw
    calibrated = max(probabilities, key=lambda level: probabilities[level] / max(prior[level], 1e-6))
    if SEVERITY[calibrated] > SEVERITY[raw] + 1:
        calibrated = next(level for level, rank in SEVERITY.items() if rank == SEVERITY[raw] + 1)
    return raw, calibrated


def benign_motif(request: TriageRequest) -> str | None:
    """Motif bénin reconnu si toutes les conditions sont réunies, sinon None.

    Conditions : un motif de la liste ; un adulte ; aucun signe d'alerte dans le texte, une fois les passages niés
    retirés ; fréquence cardiaque, pression artérielle, saturation et température toutes mesurées et normales ;
    conscience non altérée. Sans constantes complètes, la règle ne s'applique pas : le patient n'est pas réorienté.
    """
    # Champs séparés par un point : une négation d'un champ (« sans fièvre ») ne doit pas couvrir le suivant
    text = ". ".join(filter(None, (request.motif, request.symptomes, request.antecedents))).lower()
    motif = next((label for label, keywords in BENIGN_MOTIFS if any(k in text for k in keywords)), None)
    if motif is None or not BENIGN_AGE[0] <= request.age < BENIGN_AGE[1]:
        return None
    if any(word in NEGATION.sub(" ", text) for word in ALARM_WORDS):
        return None
    c = request.constantes
    for field, (low, high) in BENIGN_VITALS.items():
        value = getattr(c, field)
        if value is None or not low <= value <= high:
            return None
    for field, (low, high) in BENIGN_OPTIONAL.items():
        value = getattr(c, field)
        if value is not None and not low <= value <= high:
            return None
    if c.conscience == "alteree":
        return None
    return motif


def follow_up_questions(request: TriageRequest) -> list[str]:
    text = f"{request.motif} {request.symptomes}".lower()
    questions = []
    for keywords, motif_questions in QUESTIONS_PAR_MOTIF:
        if any(k in text for k in keywords):
            questions += [q for q in motif_questions if q not in questions]
    questions = questions or list(QUESTIONS_GENERALES)
    c = request.constantes
    questions += [q for field, q in CONSTANTES_A_MESURER.items() if getattr(c, field) is None]
    return questions


def decide(model_level: str | None, flags: list[str], benign: str | None = None) -> tuple[str, str]:
    # Niveau final et sa source. Les garde-fous ne peuvent que relever le niveau du modèle ; un motif bénin (qui
    # suppose l'absence de tout signe de gravité) donne l'urgence différée
    if flags and (model_level is None or SEVERITY[model_level] > 0):
        return "urgence maximale", "règle de sécurité"
    if benign and not flags:
        return "urgence différée", "motif bénin"
    if model_level is None:
        # Réponse du modèle inexploitable, sans signe de gravité : niveau intermédiaire, avis soignant requis
        return "urgence modérée", "défaut (réponse du modèle inexploitable)"
    return model_level, "modèle"


def justification(level: str, source: str, model_level: str | None, flags: list[str],
                  raw_level: str | None = None, benign: str | None = None) -> str:
    parts = []
    if source == "motif bénin":
        parts.append(f"Motif bénin ({benign}) chez un adulte, constantes complètes et normales, aucun signe d'alerte : "
                     "urgence différée, quelle que soit l'évaluation du modèle.")
    if flags:
        parts.append("Signes de gravité : " + " ; ".join(flags) + ".")
    if model_level and raw_level and raw_level != model_level:
        parts.append(f"Évaluation du modèle : {model_level} (avant calibration : {raw_level} ; le modèle surestime "
                     "l'urgence, son a priori est corrigé).")
    elif model_level:
        parts.append(f"Évaluation du modèle : {model_level}.")
    else:
        parts.append("Évaluation du modèle : indisponible ou inexploitable.")
    if source == "règle de sécurité":
        parts.append("Un signe de gravité impose l'urgence maximale, quelle que soit l'évaluation du modèle.")
    elif source.startswith("défaut"):
        parts.append("Par prudence, urgence modérée : un soignant doit évaluer le patient.")
    return " ".join(parts)
