
import re
import pandas as pd


def _flatten_keywords(keywords):
    if isinstance(keywords, dict):
        flattened = []
        for value in keywords.values():
            flattened.extend(_flatten_keywords(value))
        return flattened
    if isinstance(keywords, (list, tuple, set)):
        return [str(keyword) for keyword in keywords]
    return [str(keywords)]


def _keyword_pattern(keyword):
    escaped_keyword = re.escape(keyword.strip())
    return re.compile(r"(?<!\w)" + escaped_keyword + r"(?!\w)", re.IGNORECASE)


def filter_by_keywords(combined_texts, keywords):
    """
    Filters artworks based on the presence of keywords in the combined text of the artwork.

    Parameters:
    combined_texts (dict): A dictionary mapping artwork IDs to their combined text.
    keywords (dict or list): The nested keyword structure from keywords.json,
        or a flat list of keywords.

    Returns:
    dict: A dictionary containing the filtered artworks and their matching
        keywords, for example:
        {"filtered_artworks": [{"id": 1, "matched_keywords": ["street"]}]}
    """
    keyword_patterns = [
        (keyword, _keyword_pattern(keyword))
        for keyword in _flatten_keywords(keywords)
        if keyword.strip()
    ]
    artworks_df = pd.DataFrame(
        combined_texts.items(),
        columns=["id", "combined_text"],
    )

    artworks_df["matched_keywords"] = artworks_df["combined_text"].fillna("").apply(
        lambda text: [
            keyword
            for keyword, pattern in keyword_patterns
            if pattern.search(text)
        ]
    )

    filtered_artworks = artworks_df.loc[
        artworks_df["matched_keywords"].str.len() > 0,
        ["id", "matched_keywords"],
    ].to_dict(orient="records")

    return {"filtered_artworks": filtered_artworks}