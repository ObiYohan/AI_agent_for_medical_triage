"""Métadonnées du Space Hugging Face (deploy/space/README.md), vérifiées avant le déploiement : le Hub refuse le
dépôt si elles sont invalides, et l'erreur n'apparaissait qu'au job deploy-cloud."""

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "deploy" / "space" / "README.md"


def front_matter() -> dict[str, str]:
    _, header, _ = README.read_text(encoding="utf-8").split("---", 2)
    return dict(line.split(":", 1) for line in header.strip().splitlines())


def test_short_description_60_caracteres_au_plus():
    # Limite imposée par le Hub (« short_description length must be less than or equal to 60 »)
    assert len(front_matter()["short_description"].strip()) <= 60


def test_space_docker_sur_le_port_de_l_api():
    metadata = {key: value.strip() for key, value in front_matter().items()}
    assert metadata["sdk"] == "docker"
    assert metadata["app_port"] == "8080"
