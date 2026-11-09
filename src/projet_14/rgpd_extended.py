"""Anonymisation RGPD des jeux de données SFT / DPO avec Presidio.

Entrée : un fichier JSONL au format conversationnel (`messages`) ou de préférence
(`prompt` / `chosen` / `rejected`).
Sorties :
  - le JSONL anonymisé, chaque enregistrement portant un champ `anonymization` ;
  - un journal d'audit JSONL (une ligne par entité remplacée, sans la valeur d'origine) ;
  - un rapport JSON et un rapport Markdown qui documentent les paramètres et les résultats ;
  - un échantillon de contrôle manuel (avec valeurs d'origine, à ne pas versionner).

Usage :
    uv run rgpd-anonymize notebook/sft_mediqal_mcqu.jsonl --lang fr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

PIPELINE_VERSION = "1.0.0"

SPACY_MODELS = {"fr": "fr_core_news_md", "en": "en_core_web_md"}

# Entité détectée -> balise de remplacement. Les balises sont neutres (FR/EN)
# pour ne pas introduire de langue parasite dans les données d'entraînement.
REPLACEMENTS = {
    "PERSON": "<PERSON>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "DATE": "<DATE>",
    "ADDRESS": "<ADDRESS>",
    "FR_NIR": "<ID>",
    "IBAN_CODE": "<ID>",
    "CREDIT_CARD": "<ID>",
}

# Entités volontairement NON anonymisées, avec la justification reprise dans le rapport.
EXCLUDED_ENTITIES = {
    "LOCATION": "Pays et régions sont des informations cliniques (médecine des voyages, "
    "épidémiologie). Seules les adresses postales sont anonymisées.",
    "ORGANIZATION": "Les noms d'organisations (OMS, HAS, laboratoires) ne désignent pas "
    "des personnes physiques.",
    "AGE": "L'âge est indispensable au raisonnement clinique et au triage ; isolé, "
    "il ne permet pas d'identifier une personne.",
    "DURATION": "Les durées relatives (« depuis 3 jours », « J3 ») sont cliniques ; "
    "seules les dates calendaires complètes sont anonymisées.",
    "IP_ADDRESS": "Absente des textes cliniques ; le détecteur confondait des formules "
    "sanguines (79.0.0.16.4) avec des adresses IP.",
}

# Les noms propres médicaux (maladie d'Addison, épreuve de Weber...) sont reconnus
# comme des personnes par spaCy : on les écarte selon le mot qui les précède ou les suit.
EPONYM_HEADS = (
    "maladie|syndrome|signe|test|manœuvre|manoeuvre|épreuve|epreuve|score|indice|index|"
    "classification|stade|triade|tétrade|réflexe|reflexe|phénomène|phenomene|loi|cellule|"
    "corps|nodule|lymphome|sarcome|tumeur|anomalie|méthode|methode|critère|critere|"
    "échelle|echelle|temps|réaction|reaction|antigène|antigene|système|systeme|groupe|"
    "facteur|bacille|fièvre|fievre|kyste|ligament|canal|membrane|nerf|artère|artere|abcès|"
    "abces|ulcère|ulcere|anse|formule|type|point|fracture|paralysie|tétralogie|tendon|"
    "sérodiagnostic|serodiagnostic|virus|bacille|"
    "disease|sign|reflex|criterion|criteria|scale|cell|lymphoma|sarcoma|maneuver"
)
EPONYM_BEFORE = re.compile(
    rf"\b(?:{EPONYM_HEADS})[sx]?\s+(?:de\s+(?:la\s+)?|d['’]\s*|du\s+|des\s+|of\s+)?$",
    re.IGNORECASE,
)
EPONYM_AFTER = re.compile(r"^['’]s?\s+(disease|syndrome|sign|test|score|reflex|criteria|maneuver)", re.IGNORECASE)
EPONYM_CONTEXT = 40

# Sensible à la casse : « Le bilan » ou « 40 m. Il » ne doivent pas passer pour une civilité.
TITLES = (
    r"(?:[Mm]onsieur|[Mm]adame|[Mm]ademoiselle|Mme|Mlle|Melle|M\.|Mr\.?|Mrs\.?|Ms\.?|Miss|"
    r"[Dd]octeur|Dr\.?|[Pp]rofesseur|Pr\.?)"
)
TITLE_PREFIX = re.compile(rf"^{TITLES}(?:\s+|$)")

UPPER = "A-ZÀ-ÖØ-Ý"
LOWER = "a-zß-öø-ÿ"
# Forme acceptée pour un nom détecté par spaCy (hors civilité) : mot capitalisé d'au moins
# 3 lettres (Dupont, Jean-Pierre ; exclut Na, Ht, Ag) ou capitales d'au moins 5 lettres (DURAND).
# Les initiales seules (B. H.) ne sont retenues qu'après une civilité (TitleNameRecognizer).
NAME_TOKEN = re.compile(rf"^(?:[{UPPER}][{LOWER}]{{2,}}(?:[-'’][{UPPER}]?[{LOWER}]+)*|[{UPPER}]{{5,}})$")
NAME_PARTICLES = {"de", "du", "d'", "d’", "le", "la", "van", "von", "der", "ben", "el"}
LOWERCASE_WORD = re.compile(rf"\b[{LOWER}][{LOWER}\-]+\b")
# Retirés avant de construire le vocabulaire courant : « john.smith@x.com » ne doit pas
# faire de « John » un mot courant.
EMAIL_OR_URL = re.compile(r"\S+@\S+|https?://\S+|www\.\S+")

# spaCy confond noms de personnes, médicaments et éponymes : un nom n'est retenu qu'avec un
# indice narratif (âge, verbe, lien de parenté) et sans indice d'objet médical.
NAME_CUE_AFTER = re.compile(
    r"^\s*,?\s*(?:\(?\d{1,3}\s*(?:ans?|mois|semaines?|jours?|years?|months?|weeks?|days?)\b|"
    r"(?:an?\s+)?\d{1,3}[-\s](?:year|month|week|day)s?[-\s]old\b|"
    r"[âaÂA]g[ée]e?s?\b|aged\b|née?s?\b|born\b|"
    r"(?:est|a|ont|sont|était|avait|présente|consulte|souffre|vient|se\s+plaint|"
    r"is|was|has|had|presents|complains)\b)",
    re.IGNORECASE,
)
NAME_CUE_BEFORE = re.compile(
    r"\b(?:fils|fille|frère|frere|sœur|soeur|mère|mere|père|pere|papa|maman|parents|enfant|"
    r"bébé|bebe|petite?|jeune|nommée?|prénommée?|appelée?|named|called|son|daughter|brother|"
    r"sister|mother|father)\s*(?:de\s+|d['’]\s*|of\s+)?,?\s*$",
    re.IGNORECASE,
)
NOT_NAME_BEFORE = re.compile(r"(?:^|[\s(])(?:le|la|les|l['’]|du|au|aux|un|une|the|a|an)\s*$", re.IGNORECASE)
NOT_NAME_AFTER = re.compile(r"^\s*(?:®|positi|négati|negati|rose\b|latéralis|lateralis)", re.IGNORECASE)

MONTHS_FR = "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
MONTHS_EN = "january|february|march|april|may|june|july|august|september|october|november|december"


class TitleNameRecognizer(EntityRecognizer):
    """Nom ou initiales précédés d'une civilité : « Madame F. », « Dr Martin », « Mr R. S. ».

    Seul le nom est retenu : la civilité (et donc le sexe, information clinique) est conservée.
    """

    # Initiales : « X », « R. », « B. H. », « M.H », « P... », « B-M » ;
    # noms : « Dupont », « Jean Martin », « DURAND ».
    _NAME = rf"[{UPPER}][{LOWER}]+(?:[-'’][{UPPER}]?[{LOWER}]+)*"
    PATTERN = re.compile(
        rf"\b{TITLES}\s+("
        rf"[{UPPER}](?:-[{UPPER}])+\b"
        rf"|(?:[{UPPER}](?:\.{{3}}|…|\.|\b)\s?){{1,3}}"
        rf"|{_NAME}(?:\s+{_NAME})?"
        rf"|[{UPPER}]{{2,}}(?:-[{UPPER}]+)*\b)"
    )
    # « produit M. La justification » : « M. » termine une phrase, ce n'est pas une civilité.
    NOT_NAMES = {
        "La", "Le", "Les", "Il", "Ils", "Elle", "Elles", "Un", "Une", "Ce", "Cet", "Cette", "Ces",
        "On", "En", "De", "Des", "Du", "Au", "Aux", "Son", "Sa", "Ses", "Dans", "Pour", "Par", "Sur",
        "Chez", "Lors", "Après", "Avant", "Quel", "Quelle", "The", "He", "She", "This", "It", "In",
    }

    def __init__(self, supported_language: str):
        super().__init__(supported_entities=["PERSON"], supported_language=supported_language, name="TitleNameRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for match in self.PATTERN.finditer(text):
            start, end = match.span(1)
            if match.group(1).split()[0] in self.NOT_NAMES:
                continue
            while end > start and text[end - 1].isspace():
                end -= 1
            results.append(
                RecognizerResult(
                    "PERSON",
                    start,
                    end,
                    0.9,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results


def custom_recognizers(lang: str) -> list[EntityRecognizer]:
    """Détecteurs pour les identifiants absents de Presidio ou mal adaptés au domaine médical."""
    dates = PatternRecognizer(
        supported_entity="DATE",
        supported_language=lang,
        name="FullDateRecognizer",
        # Année obligatoire : évite de confondre une tension (12/8) ou un ratio avec une date.
        patterns=[
            Pattern("date_numerique", r"\b(0?[1-9]|[12]\d|3[01])[/.\-](0?[1-9]|1[0-2])[/.\-](19|20)\d{2}\b", 0.7),
            # Année sur 2 chiffres (20/12/97) : barres obliques uniquement, trois composantes.
            Pattern("date_numerique_courte", r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/\d{2}\b", 0.6),
            Pattern("date_fr", rf"\b(1er|[12]?\d|3[01])\s+({MONTHS_FR})\s+(19|20)\d{{2}}\b", 0.7),
            Pattern("date_en", rf"\b({MONTHS_EN})\s+\d{{1,2}},?\s+(19|20)\d{{2}}\b", 0.7),
            Pattern("date_en_jour", rf"\b\d{{1,2}}\s+({MONTHS_EN})\s+(19|20)\d{{2}}\b", 0.7),
        ],
    )
    nir = PatternRecognizer(
        supported_entity="FR_NIR",
        supported_language=lang,
        name="FrNirRecognizer",
        patterns=[
            Pattern(
                "nir",
                r"\b[12][\s.]?\d{2}[\s.]?(0[1-9]|1[0-2]|[2-9]\d)[\s.]?(\d{2}|2[AB])[\s.]?\d{3}[\s.]?\d{3}(?:[\s.]?\d{2})?\b",
                0.6,
            )
        ],
        context=["sécurité sociale", "securite sociale", "nir", "sécu", "secu", "ins"],
    )
    address = PatternRecognizer(
        supported_entity="ADDRESS",
        supported_language=lang,
        name="AddressRecognizer",
        patterns=[
            Pattern(
                "adresse_fr",
                r"\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|chemin|allée|allee|"
                r"impasse|place|route|quai|cours)\s+[^\n,;.]{2,60}",
                0.6,
            ),
            Pattern(
                "adresse_en",
                r"\b\d{1,5}\s+[A-Za-z][\w ]{1,40}\s(?:street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd\.|lane|drive)\b",
                0.6,
            ),
        ],
    )
    return [TitleNameRecognizer(lang), dates, nir, address]


def build_analyzer(languages: list[str]) -> AnalyzerEngine:
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": SPACY_MODELS[lang]} for lang in languages],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {"PER": "PERSON", "PERSON": "PERSON"},
            "labels_to_ignore": ["MISC", "LOC", "GPE", "ORG", "NORP", "DATE", "TIME", "CARDINAL"],
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
    registry = RecognizerRegistry(supported_languages=languages)
    for lang in languages:
        registry.add_recognizer(SpacyRecognizer(supported_language=lang, supported_entities=["PERSON"]))
        registry.add_recognizer(EmailRecognizer(supported_language=lang))
        registry.add_recognizer(
            PhoneRecognizer(supported_language=lang, supported_regions=("FR", "BE", "CH", "GB", "US", "CA"))
        )
        registry.add_recognizer(IbanRecognizer(supported_language=lang))
        registry.add_recognizer(CreditCardRecognizer(supported_language=lang))
        for recognizer in custom_recognizers(lang):
            registry.add_recognizer(recognizer)
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=languages)


def is_eponym(text: str, result: RecognizerResult) -> bool:
    before = text[max(0, result.start - EPONYM_CONTEXT) : result.start]
    after = text[result.end : result.end + EPONYM_CONTEXT]
    return bool(EPONYM_BEFORE.search(before) or EPONYM_AFTER.search(after))


def trim(text: str, result: RecognizerResult) -> RecognizerResult | None:
    """Retire la civilité en tête et les espaces / ponctuation en fin d'entité."""
    start, end = result.start, result.end
    match = TITLE_PREFIX.match(text[start:end])
    if match:
        start += match.end()
    while end > start and (text[end - 1].isspace() or text[end - 1] in ":;,"):
        end -= 1
    if start >= end:
        return None
    return RecognizerResult(
        result.entity_type, start, end, result.score, result.analysis_explanation, result.recognition_metadata
    )


def name_rejection(text: str, result: RecognizerResult, common_words: set[str]) -> str | None:
    """Raison du rejet d'un nom détecté par spaCy, ou None s'il est retenu."""
    span = text[result.start : result.end]
    name_tokens = [t for t in span.split() if t.lower() not in NAME_PARTICLES]
    if not name_tokens or not all(NAME_TOKEN.match(t) for t in name_tokens):
        return "forme non nominale"
    # Un mot qui apparaît aussi en minuscules dans le corpus est un nom commun
    # (Fibrinogène, Abdomen, Névrose en début de ligne).
    if all(t.lower() in common_words for t in name_tokens):
        return "mot courant du corpus"
    before = text[max(0, result.start - EPONYM_CONTEXT) : result.start]
    after = text[result.end : result.end + EPONYM_CONTEXT]
    if NOT_NAME_BEFORE.search(before) or NOT_NAME_AFTER.search(after):
        return "contexte d'objet médical"
    if not (NAME_CUE_BEFORE.search(before) or NAME_CUE_AFTER.search(after)):
        return "aucun indice de personne"
    return None


def propagate(text: str, results: list[RecognizerResult], names: set[str]) -> list[RecognizerResult]:
    """Ajoute toutes les occurrences des noms déjà confirmés dans l'enregistrement."""
    extra = []
    # Les noms complets d'abord : en cas de chevauchement, remove_overlaps garde le plus long.
    for name in sorted(names, key=len, reverse=True):
        for match in re.finditer(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
            extra.append(
                RecognizerResult(
                    "PERSON",
                    match.start(),
                    match.end(),
                    0.8,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: "RecordPropagation",
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: "RecordPropagation",
                    },
                )
            )
    return results + extra


def remove_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Garde, en cas de chevauchement, l'entité au score le plus élevé puis la plus longue."""
    kept: list[RecognizerResult] = []
    for result in sorted(results, key=lambda r: (-r.score, -(r.end - r.start))):
        if all(result.end <= k.start or result.start >= k.end for k in kept):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


class Anonymizer:
    def __init__(self, languages: list[str], threshold: float, common_words: set[str]):
        self.analyzer = build_analyzer(languages)
        self.anonymizer = AnonymizerEngine()
        self.threshold = threshold
        self.common_words = common_words
        self.entities = list(REPLACEMENTS)
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": tag}) for entity, tag in REPLACEMENTS.items()
        }
        self.discarded = Counter()
        # Candidats écartés (texte, début, fin, raison), pour le contrôle manuel du rappel.
        self.rejected: list[tuple[str, int, int, str]] = []

    def detect(self, text: str, lang: str, names: set[str]) -> list[RecognizerResult]:
        """Détecte les entités ; `names` accumule les noms confirmés de l'enregistrement."""
        results = []
        for result in self.analyzer.analyze(text, language=lang, entities=self.entities, score_threshold=self.threshold):
            if result.recognition_metadata.get("recognizer_name") == "SpacyRecognizer":
                result = trim(text, result)
                if result is None:
                    self.discarded["PERSON (civilité seule)"] += 1
                    continue
                reason = "éponyme médical" if is_eponym(text, result) else name_rejection(text, result, self.common_words)
                if reason:
                    self.discarded[f"PERSON ({reason})"] += 1
                    self.rejected.append((text, result.start, result.end, reason))
                    continue
            results.append(result)
        for r in results:
            if r.entity_type != "PERSON":
                continue
            # Nom complet et chacune de ses parties (« John Smith », « John », « Smith »).
            # Les initiales (R., B. H.) ne sont pas propagées : trop ambiguës hors civilité.
            tokens = [t for t in text[r.start : r.end].split() if NAME_TOKEN.match(t)]
            if tokens:
                names.update(tokens)
                names.add(" ".join(tokens))
        return remove_overlaps(propagate(text, results, names))

    def anonymize(self, text: str, lang: str, names: set[str]) -> tuple[str, list[RecognizerResult]]:
        results = self.detect(text, lang, names)
        if not results:
            return text, []
        return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=self.operators).text, results


def text_fields(record: dict):
    """Liste les champs texte à anonymiser : (chemin, conteneur, clé).

    Le message `system` est un gabarit commun sans donnée personnelle : il est ignoré.
    """

    def message_fields(prefix: str, messages: list[dict]):
        for i, message in enumerate(messages):
            if message.get("role") != "system" and isinstance(message.get("content"), str):
                yield f"{prefix}[{i}].content", message, "content"

    if isinstance(record.get("messages"), list):
        yield from message_fields("messages", record["messages"])
    for key in ("prompt", "chosen", "rejected"):
        value = record.get(key)
        if isinstance(value, str):
            yield key, record, key
        elif isinstance(value, list):
            yield from message_fields(key, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
    default_lang: str,
    threshold: float,
    review_sample: int,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_path = output_dir / f"{stem}.anonymized.jsonl"
    audit_path = report_dir / f"{stem}.audit.jsonl"

    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    languages = sorted({r.get("lang", default_lang) for r in records})
    common_words = {
        word
        for record in records
        for _, container, key in text_fields(record)
        for word in LOWERCASE_WORD.findall(EMAIL_OR_URL.sub(" ", container[key]))
    }
    engine = Anonymizer(languages, threshold, common_words)

    by_entity, by_recognizer, by_lang = Counter(), Counter(), Counter()
    modified_records = 0
    review_candidates = []

    with output_path.open("w", encoding="utf-8") as out, audit_path.open("w", encoding="utf-8") as audit:
        for index, record in enumerate(records):
            record_id = str(record.get("id", index))
            lang = record.get("lang", default_lang)
            by_lang[lang] += 1
            record_entities = Counter()
            names: set[str] = set()
            for field, container, key in list(text_fields(record)):
                original = container[key]
                anonymized, results = engine.anonymize(original, lang, names)
                for r in results:
                    recognizer = r.recognition_metadata.get("recognizer_name", "unknown")
                    by_entity[r.entity_type] += 1
                    by_recognizer[recognizer] += 1
                    record_entities[r.entity_type] += 1
                    # La valeur d'origine n'est jamais écrite dans le journal d'audit.
                    audit.write(
                        json.dumps(
                            {
                                "record_id": record_id,
                                "field": field,
                                "entity_type": r.entity_type,
                                "start": r.start,
                                "end": r.end,
                                "length": r.end - r.start,
                                "score": round(r.score, 3),
                                "recognizer": recognizer,
                                "operator": "replace",
                                "new_value": REPLACEMENTS[r.entity_type],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    review_candidates.append((record_id, field, original, r))
                container[key] = anonymized
            if record_entities:
                modified_records += 1
            record["anonymization"] = {
                "pipeline_version": PIPELINE_VERSION,
                "anonymized": True,
                "entities_replaced": dict(record_entities),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    review_path = None
    if review_sample and (review_candidates or engine.rejected):
        review_path = report_dir / f"{stem}.review.txt"
        rng = random.Random(seed)

        def context(text: str, start: int, end: int) -> str:
            left = text[max(0, start - 60) : start].replace("\n", " ")
            right = text[end : end + 60].replace("\n", " ")
            return f"…{left}⟦{text[start:end]}⟧{right}…"

        with review_path.open("w", encoding="utf-8") as f:
            f.write("# Échantillon de contrôle manuel : CONTIENT LES VALEURS D'ORIGINE, NE PAS VERSIONNER.\n\n")
            f.write("## Entités remplacées (vérifier la précision : faux positifs ?)\n\n")
            for record_id, field, original, r in rng.sample(review_candidates, min(review_sample, len(review_candidates))):
                f.write(f"[{record_id} | {field} | {r.entity_type} | {r.score:.2f}]\n{context(original, r.start, r.end)}\n\n")
            f.write("## Candidats PERSON écartés (vérifier le rappel : vrais noms manqués ?)\n\n")
            for text, start, end, reason in rng.sample(engine.rejected, min(review_sample, len(engine.rejected))):
                f.write(f"[{reason}]\n{context(text, start, end)}\n\n")

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"path": input_path.as_posix(), "sha256": sha256(input_path), "records": len(records)},
        "output": {"path": output_path.as_posix(), "sha256": sha256(output_path)},
        "audit_log": audit_path.as_posix(),
        "review_sample": review_path.as_posix() if review_path else None,
        "parameters": {
            "languages": languages,
            "spacy_models": {lang: SPACY_MODELS[lang] for lang in languages},
            "score_threshold": threshold,
            "seed": seed,
            "replacements": REPLACEMENTS,
            "excluded_entities": EXCLUDED_ENTITIES,
            "fields": "messages[*].content (hors system), prompt, chosen, rejected",
        },
        "versions": {pkg: version(pkg) for pkg in ("presidio-analyzer", "presidio-anonymizer", "spacy")},
        "results": {
            "records_by_lang": dict(by_lang),
            "records_modified": modified_records,
            "entities_replaced": dict(by_entity.most_common()),
            "entities_by_recognizer": dict(by_recognizer.most_common()),
            "detections_discarded": dict(engine.discarded.most_common()),
        },
    }
    (report_dir / f"{stem}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / f"{stem}.report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    p, res = report["parameters"], report["results"]

    def rows(d: dict) -> str:
        return "\n".join(f"| {k} | {v} |" for k, v in d.items()) or "| aucune | 0 |"

    return f"""# Rapport d'anonymisation RGPD

- **Date (UTC)** : {report['date_utc']}
- **Version du pipeline** : {report['pipeline_version']}
- **Fichier source** : `{report['input']['path']}` ({report['input']['records']} enregistrements)
- **Fichier produit** : `{report['output']['path']}`
- **Journal d'audit** : `{report['audit_log']}` (une ligne par entité remplacée, sans valeur d'origine)

## Méthode

Détection avec Presidio {report['versions']['presidio-analyzer']} et spaCy {report['versions']['spacy']}
(modèles : {', '.join(f'`{m}`' for m in p['spacy_models'].values())}), seuil de score {p['score_threshold']}.
Chaque entité détectée est remplacée par une balise neutre. Aucune table de correspondance n'est
conservée : le remplacement est irréversible.

Champs traités : {p['fields']}.

### Entités remplacées

| Entité | Balise |
|---|---|
{rows(p['replacements'])}

### Entités conservées volontairement

| Entité | Justification |
|---|---|
{rows(p['excluded_entities'])}

### Règles propres au domaine médical

- Les éponymes médicaux (maladie d'Addison, épreuve de Weber, Addison's disease...) détectés comme
  des personnes sont écartés selon leur contexte.
- La civilité est conservée et seul le nom est remplacé (« Madame <PERSON> »), car le sexe est une
  information clinique. Les initiales ne sont remplacées qu'après une civilité.
- Un nom détecté par spaCy n'est retenu que s'il a une forme de nom propre (mot capitalisé d'au
  moins 3 lettres, ou capitales d'au moins 5 lettres), qu'il n'apparaît pas en minuscules ailleurs
  dans le corpus (« Fibrinogène » en début de ligne est un nom commun), qu'il n'est pas dans un
  contexte d'objet médical (« de la Nivaquine® », « le Rinne est négatif ») et qu'un indice
  narratif le désigne comme une personne (« Ahmed, âgé de 12 ans », « Denis est examiné »,
  « leur fille, Jacqueline »).
- Un nom confirmé est remplacé dans toutes ses occurrences de l'enregistrement.
- Seules les dates calendaires avec année sont remplacées : les tensions (12/8) et durées sont conservées.

## Résultats

- **Enregistrements par langue** : {res['records_by_lang']}
- **Enregistrements modifiés** : {res['records_modified']} / {report['input']['records']}

| Entité | Remplacements |
|---|---|
{rows(res['entities_replaced'])}

| Détecteur | Détections |
|---|---|
{rows(res['entities_by_recognizer'])}

| Détections écartées | Nombre |
|---|---|
{rows(res['detections_discarded'])}

## Intégrité

| Fichier | SHA-256 |
|---|---|
| Source | `{report['input']['sha256']}` |
| Produit | `{report['output']['sha256']}` |

## Limites

- La détection automatique n'est pas exhaustive : un contrôle manuel sur échantillon est requis
  (fichier `review`, qui contient les valeurs d'origine et ne doit pas être versionné).
- Les règles privilégient la précision (ne pas altérer le vocabulaire médical) : un nom sans
  indice narratif, ou identique à un mot courant (Rose), peut être manqué. Pour des données de
  patients réels, ajouter un lexique de prénoms (fichier des prénoms INSEE) et relever le rappel.
- Les quasi-identifiants (âge, profession, lieu) sont conservés pour leur valeur clinique ; leur
  combinaison doit être réévaluée avant tout usage de données de patients réels.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymisation RGPD d'un dataset JSONL (SFT / DPO) avec Presidio.")
    parser.add_argument("input", type=Path, help="Fichier JSONL à anonymiser")
    parser.add_argument("--output-dir", type=Path, default=Path("data/anonymized"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/rgpd"))
    parser.add_argument(
        "--lang", default="fr", choices=sorted(SPACY_MODELS), help="Langue si l'enregistrement n'a pas de champ `lang`"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35, help="Score minimal de détection (0.4 pour les téléphones)"
    )
    parser.add_argument(
        "--review-sample", type=int, default=50, help="Détections extraites pour contrôle manuel (0 : désactivé)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run(args.input, args.output_dir, args.report_dir, args.lang, args.threshold, args.review_sample, args.seed)
    res = report["results"]
    print(f"{res['records_modified']} / {report['input']['records']} enregistrements modifiés")
    print(f"Entités remplacées : {res['entities_replaced']}")
    print(f"Détections écartées : {res['detections_discarded']}")
    print(f"Rapport : {args.report_dir / (args.input.stem + '.report.md')}")


if __name__ == "__main__":
    main()
