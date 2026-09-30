"""
Use a NER model to extract named entities from the text. used after filtering the artworks by keywords. The named entities can be used to geolocate the artworks.
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
    ner_pipeline = pipeline("ner", 
                            model=model, 
                            tokenizer=tokenizer, 
                            aggregation_strategy="simple",
                            device=0  # Use GPU if available, otherwise use CPU
    )
                            
    return ner_pipeline

def extract_named_entities(filtered_texts, ner_pipeline):
    """
    Extract named entities from the text using the provided NER pipeline.

    Parameters:
    filtered_texts (dict): A dictionary mapping artwork IDs to their text.
    ner_pipeline (pipeline): A Hugging Face pipeline for named entity recognition.

    Returns:
    dict: A dictionary mapping artwork IDs to their extracted named entities.
    """
    texts = list(filtered_texts.values())

    results = ner_pipeline(texts)
    return dict(zip(filtered_texts.keys(), results))

def extract_entities_from_artworks(artworks, ner_pipeline):
    """
    Extract named entities from the combined text of artworks.

    Parameters:
    artworks (dict): A dictionary mapping artwork IDs to their combined text.
    ner_pipeline (pipeline): A Hugging Face pipeline for named entity recognition.

    Returns:
    dict: A dictionary mapping artwork IDs to their extracted named entities.
    """
    texts = utils.combine_all_artworks_text_fields(artworks)
    
    results = {}
    results = ner_pipeline(texts)

    return results


def display_named_entities(entities):
    """
    Display the named entities with id and identified entity.

    param entities (dict): A dictionary mapping artwork IDs to their extracted named entities.
    """

    for artwork_id, entity_list in entities.items():
        print(f"Artwork ID: {artwork_id}")
        for entity in entity_list:
            print(f"  Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}")
        print()