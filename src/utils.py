import json
from pathlib import Path

def combine_artwork_text_fields(artwork):
    text_fields = [
        artwork.get("title", ""),
        artwork.get("alt_text", ""),
        artwork.get("description", ""),
        artwork.get("short_description", ""),
        " ".join(artwork.get("subject_titles", [])),
        " ".join(artwork.get("theme_titles", [])),
        " ".join(artwork.get("style_titles", [])),
    ]
    return " ".join(str(field) for field in text_fields if field)


def read_artworks_from_json(json_file):
    with Path(json_file).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        artwork_records = payload.get("unique_artworks", payload.get("data", []))
    else:
        artwork_records = payload

    artworks = {}
    for record in artwork_records:
        artwork = record.get("artwork", record)
        artwork_id = artwork.get("id")
        if artwork_id is not None:
            artworks[artwork_id] = artwork
    return artworks


def combine_all_artworks_text_fields(artworks):
    combined_texts = {}
    for artwork_record in artworks.get("unique_artworks", []):
        artwork = artwork_record.get("artwork", artwork_record)
        artwork_id = artwork.get("id")
        combined_texts[artwork_id] = combine_artwork_text_fields(artwork)
    return combined_texts


def read_all_json_files_and_combine_text_fields(raw_aic_dir):
    combined_texts = {}
    for json_file in sorted(Path(raw_aic_dir).glob("*.json")):
        with json_file.open("r", encoding="utf-8") as file:
            combined_texts.update(
                combine_all_artworks_text_fields(json.load(file))
            )
    return combined_texts