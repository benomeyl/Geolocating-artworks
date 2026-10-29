"""
NER helpers of the early exploration pipeline (used by src/main.py).

The current pipeline loads its NER in run_pipeline.load_ner_pipeline and
extracts candidates with filtering_logreg.extract_candidate_artwork.
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    from src import utils
except ImportError:  # pragma: no cover - fallback for script execution
    import utils

def load_ner_model(model_name="Babelscape/wikineural-multilingual-ner"):
    """
    Import a NER model from Hugging Face.

    Parameters:
    model_name (str): The name of the model to import. Default is "Babelscape/wikineural-multilingual-ner".

    Returns:
    pipeline: A Hugging Face pipeline for named entity recognition.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",  # Avoids split sub-word fragments ("##kirche").
        device=0,  # First GPU.
    )


def extract_named_entities(filtered_texts, ner_pipeline):
    """
    Extract named entities from the text using the provided NER pipeline.

    Parameters:
    filtered_texts (dict): A dictionary mapping artwork IDs to their text.
    ner_pipeline (pipeline): A Hugging Face pipeline for named entity recognition.

    Returns:
    dict: A dictionary mapping artwork IDs to their extracted named entities.
    """
    results = ner_pipeline(list(filtered_texts.values()))
    return dict(zip(filtered_texts.keys(), results))


def extract_entities_from_artworks(artworks, ner_pipeline):
    """
    Extract named entities from the combined text of artworks.

    Parameters:
    artworks (dict): A corpus payload with a "unique_artworks" list.
    ner_pipeline (pipeline): A Hugging Face pipeline for named entity recognition.

    Returns:
    The pipeline output for the combined texts (see utils.combine_all_artworks_text_fields).
    """
    texts = utils.combine_all_artworks_text_fields(artworks)
    return ner_pipeline(texts)


def display_named_entities(entities):
    """
    Print the named entities of each artwork.

    Parameters:
    entities (dict): A dictionary mapping artwork IDs to their extracted named entities.
    """

    for artwork_id, entity_list in entities.items():
        print(f"Artwork ID: {artwork_id}")
        for entity in entity_list:
            print(f"  Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}")
        print()