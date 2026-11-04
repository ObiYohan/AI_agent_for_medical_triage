"""Anonymisation RGPD simplifiée : remplace les noms de personnes par <PERSON>.

Version volontairement simple de rgpd_extended.py : une seule entité (PERSON), une seule langue,
et trois règles pour écarter les faux positifs les plus fréquents (éponymes médicaux, sigles,
mots sans majuscule).

Usage en ligne de commande :
    uv run rgpd-anonymize notebook/sft_mediqal_mcqu.jsonl --lang fr

Usage depuis Python (notebook) :
    from projet_14.rgpd import run
    report = run("sft_mediqal_mcqu.jsonl", lang="fr", output_dir="../data/anonymized")
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

SPACY_MODELS = {"fr": "fr_core_news_md", "en": "en_core_web_md"}

# Chaque nom détecté est remplacé par cette balise.
OPERATORS = {"PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"})}

# Civilités : elles sont conservées (le sexe est une information clinique), seul le nom est remplacé.
TITLES = ["Monsieur", "Madame", "Mademoiselle", "Mme", "Mlle", "M.", "Mr", "Mrs", "Docteur", "Dr"]

# Nom ou initiale juste après une civilité : « Monsieur R. », « Mme Dupont ».
# Un lookbehind par civilité, car Python n'accepte que des lookbehinds de longueur fixe.
TITLE_NAME = "(?:" + "|".join(rf"(?<={re.escape(t)} )" for t in TITLES) + r")[A-ZÀ-Ý](?:[a-zà-ÿ]+|\.)"

# Éponymes médicaux : « maladie d'Addison », « épreuve de Weber », « temps de Quick », « Addison's disease ».
# Mots qui, placés devant un nom propre, en font un objet médical et non une personne.
EPONYM_HEADS = (
    "maladie|syndrome|signe|test|épreuve|score|classification|réflexe|temps|critère|cellule|"
    "réaction|phénomène|manœuvre|méthode|échelle|indice|index|stade|triade|loi|type|point|"
    "fracture|paralysie|lymphome|sarcome|tumeur|kyste|bacille|virus|ligament|canal|nerf|"
    "disease|sign|criteria|cell|scale|reflex"
)
EPONYM_BEFORE = re.compile(
    rf"\b(?:{EPONYM_HEADS})[sx]?\s+(?:de\s+(?:la\s+)?|d['’]\s*|du\s+|des\s+|of\s+)?$",
    re.IGNORECASE,
)
EPONYM_AFTER = re.compile(r"^['’]s?\s+(?:disease|syndrome|sign|test)", re.IGNORECASE)

# Mots détectés à tort comme des personnes (médicaments, examens, termes médicaux), en minuscules.
# À compléter au fil des contrôles manuels.
EXCLUDED_WORDS = {
    "nivaquine", "glibenclamide", "artane", "aldomet",  # médicaments
    "fibrinogène", "na", "céto",  # biologie
    "osseuses", "schuller",  # termes médicaux
}


def build_analyzer(lang: str) -> AnalyzerEngine:
    """Analyseur Presidio qui ne cherche que des PERSON, dans une seule langue."""
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": SPACY_MODELS[lang]}],
        }
    ).create_engine()

    registry = RecognizerRegistry(supported_languages=[lang])
    # Détection des noms par le modèle spaCy (NER).
    registry.add_recognizer(SpacyRecognizer(supported_language=lang, supported_entities=["PERSON"]))
    # Détection des noms après une civilité, par expression régulière.
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PERSON",
            supported_language=lang,
            name="TitleNameRecognizer",
            patterns=[Pattern("civilite_nom", TITLE_NAME, 0.9)],
        )
    )
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=[lang])


def strip_title(text: str, result) -> None:
    """Retire la civilité en tête de l'entité : « Monsieur Daniel » -> « Daniel »."""
    for title in TITLES:
        if text[result.start : result.end].startswith(title + " "):
            result.start += len(title) + 1
            return


def rejection_reason(text: str, result) -> str | None:
    """Raison d'écarter une détection, ou None si c'est bien un nom de personne."""
    span = text[result.start : result.end]
    if not span or not span[0].isupper():
        return "pas de majuscule"
    if span.isalpha() and span.isupper() and len(span) <= 4:
        return "sigle"  # ECG, RAA, VIH...
    if span.lower() in EXCLUDED_WORDS:
        return "mot exclu"
    before = text[max(0, result.start - 30) : result.start]
    after = text[result.end : result.end + 30]
    if EPONYM_BEFORE.search(before) or EPONYM_AFTER.search(after):
        return "éponyme médical"
    if span.endswith("®") or after.lstrip().startswith("®"):
        return "marque déposée"  # Aldomet®, Sectral ®
    return None


def anonymize_text(text, lang, analyzer, anonymizer, threshold, discarded: Counter) -> tuple[str, int]:
    """Renvoie le texte anonymisé et le nombre de noms remplacés."""
    kept = []
    for result in analyzer.analyze(text=text, entities=["PERSON"], language=lang, score_threshold=threshold):
        strip_title(text, result)
        reason = rejection_reason(text, result)
        if reason:
            discarded[reason] += 1
        else:
            kept.append(result)
    if not kept:
        return text, 0
    return anonymizer.anonymize(text=text, analyzer_results=kept, operators=OPERATORS).text, len(kept)


def run(
    input_path: Path, lang: str = "fr", threshold: float = 0.5, output_dir: Path = Path("data/anonymized")
) -> dict:
    """Anonymise un fichier JSONL, écrit le fichier anonymisé et le rapport, et renvoie le rapport."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    analyzer = build_analyzer(lang)
    anonymizer = AnonymizerEngine()
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    replaced, modified_records, discarded = 0, 0, Counter()
    for record in records:
        record_count = 0
        # Le message system est un gabarit commun, sans donnée personnelle.
        for message in record["messages"]:
            if message["role"] != "system":
                message["content"], count = anonymize_text(
                    message["content"], lang, analyzer, anonymizer, threshold, discarded
                )
                record_count += count
        record["persons_replaced"] = record_count
        replaced += record_count
        modified_records += record_count > 0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.anonymized.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "lang": lang,
        "spacy_model": SPACY_MODELS[lang],
        "threshold": threshold,
        "records": len(records),
        "records_modified": modified_records,
        "persons_replaced": replaced,
        "detections_discarded": dict(discarded),
    }
    (output_dir / f"{input_path.stem}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymisation simplifiée des noms de personnes d'un JSONL.")
    parser.add_argument("input", type=Path, help="Fichier JSONL au format `messages`")
    parser.add_argument("--lang", default="fr", choices=sorted(SPACY_MODELS))
    parser.add_argument("--threshold", type=float, default=0.5, help="Score minimal de détection")
    parser.add_argument("--output-dir", type=Path, default=Path("data/anonymized"))
    args = parser.parse_args()

    report = run(args.input, args.lang, args.threshold, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
