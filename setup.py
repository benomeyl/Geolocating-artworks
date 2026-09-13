from pathlib import Path


PROJECT_NAME = "geolocating-artworks"


DIRECTORIES = [
    "data/raw/aic",
    "data/processed",
    "src/api",
    "src/data",
    "src/geolocation",
    "notebooks",
    "tests",
    "results",
]


FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "data/README.md",
    "src/__init__.py",
    "src/main.py",
    "src/api/__init__.py",
    "src/api/aic.py",
    "src/data/__init__.py",
    "src/data/filtering.py",
    "src/geolocation/__init__.py",
    "tests/.gitkeep",
]


FILE_CONTENTS = {
    ".gitignore": """\
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/

data/raw/
data/processed/
results/

.pytest_cache/
.mypy_cache/
.ruff_cache/
""",

    "requirements.txt": """\
requests
pandas
""",

    "README.md": """\
# Geolocating Artworks

EPFL project for geolocating artworks from open museum collections.

## Project structure

- `data/raw/` : données brutes récupérées depuis les sources
- `data/processed/` : données nettoyées et préparées
- `src/api/` : communication avec les APIs
- `src/data/` : traitement et filtrage des données
- `src/geolocation/` : géolocalisation et modèles
- `notebooks/` : exploration et expérimentations
- `tests/` : tests
- `results/` : résultats des expériences
""",

    "pyproject.toml": """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "geolocating-artworks"
version = "0.1.0"
description = "EPFL project for geolocating artworks"
requires-python = ">=3.10"
dependencies = [
    "requests",
    "pandas",
]
""",

    "data/README.md": """\
# Data

Les données sont organisées en deux catégories :

- `raw/` : données originales récupérées depuis les APIs
- `processed/` : données nettoyées et transformées
""",

    "src/main.py": """\
def main():
    print("Geolocating Artworks")


if __name__ == "__main__":
    main()
""",

    "src/api/aic.py": """\
# Client pour l'API de l'Art Institute of Chicago.
""",

    "src/data/filtering.py": """\
# Fonctions de filtrage et de nettoyage des données.
""",
}


def create_project():
    project_path = Path(PROJECT_NAME)

    if project_path.exists():
        print(f"Erreur : le dossier '{PROJECT_NAME}' existe déjà.")
        return

    project_path.mkdir()

    for directory in DIRECTORIES:
        path = project_path / directory
        path.mkdir(parents=True)

    for file in FILES:
        path = project_path / file

        content = FILE_CONTENTS.get(file, "")
        path.write_text(content, encoding="utf-8")

    print(f"Projet créé dans : {project_path.resolve()}")
    print()
    print("Structure créée :")

    for path in sorted(project_path.rglob("*")):
        relative_path = path.relative_to(project_path)

        if path.is_dir():
            print(f"  {relative_path}/")
        else:
            print(f"  {relative_path}")


if __name__ == "__main__":
    create_project()