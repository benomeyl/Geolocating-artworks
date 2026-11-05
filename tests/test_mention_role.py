from src.data.filtering_logreg import mention_role_features, portrait_features


def _occurrence(text, place, field="title", start=None):
    start = text.index(place) if start is None else start
    return {"source_text": text, "start": start, "end": start + len(place), "field": field, "place": place}


def _roles(*occurrences):
    return {name for name, value in mention_role_features(list(occurrences)).items() if value}


def test_depiction_and_origin_cues_before_the_mention():
    assert _roles(_occurrence("View of Delft", "Delft")) == {"role_depiction_cue"}
    assert _roles(_occurrence("<p>The print was printed in London by Boydell.</p>", "London", "description")) == {
        "role_origin_cue"
    }
    # An origin verb is not counted as a plain location preposition.
    assert "role_location_preposition" not in _roles(_occurrence("Artist born in Leiden", "Leiden", "description"))


def test_position_in_the_title():
    title = "Le Pont Neuf, Paris"
    assert _roles(_occurrence(title, "Pont Neuf")) == {"role_title_head"}
    assert _roles(_occurrence(title, "Paris")) == {"role_title_after_comma"}
    assert _roles(_occurrence("Brooklyn Bridge from Brooklyn", "Brooklyn", start=21)) == {"role_from"}


def test_place_names_inside_institution_names():
    assert "role_institution_name" in _roles(_occurrence("Cover of the Chicago Tribune", "Chicago"))
    assert "role_institution_name" in _roles(_occurrence("Lecture at the University of Chicago", "Chicago"))
    assert "role_institution_name" not in _roles(_occurrence("Chicago River", "Chicago"))


def test_any_occurrence_sets_the_feature():
    roles = _roles(_occurrence("Venice", "Venice"), _occurrence("Painted in Venice", "Venice", "description"))
    assert roles == {"role_title_head", "role_location_preposition"}


def test_portrait_clues():
    title = "Alfred Stieglitz and Georgia O'Keeffe, New York City"
    occurrence = {**_occurrence(title, "New York City"), "title_person_count": 2}
    assert portrait_features([occurrence], {"subject_titles": ["Portraits", "women"]}) == {
        "title_names_person": 1, "subject_people": 1, "title_place_suffix": 1,
    }
    bridge = _occurrence("Annunciation Bridge in St. Petersburg", "St. Petersburg")
    assert portrait_features([bridge], {"subject_titles": ["Bridges"]}) == {
        "title_names_person": 0, "subject_people": 0, "title_place_suffix": 1,
    }
    assert portrait_features([_occurrence("Paris Street; Rainy Day", "Paris")], {})["title_place_suffix"] == 0
