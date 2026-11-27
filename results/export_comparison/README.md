# Gold standard and predictions of the metadata-only model — comparison package

This package contains the annotated artworks (gold standard) of the
metadata-only approach and, for each artwork, the prediction of two of our
models. Its purpose is to compare this approach with an image-based approach
on exactly the same artworks.

Files: `gold_and_predictions.csv` (one row per artwork) and
`gold_and_predictions.json` (same content).

## Task

For a given artwork, find the **real place it depicts** (not where it was
made, nor the city of the artist or of a portrait's sitter), at the most
specific level possible: exact place (monument, building, street, natural
site), neighbourhood, city, region or country. An answer at a broader level
is correct if it contains the annotated place ("Paris" is correct at city
level for the Pont Neuf). Abstaining is allowed and preferable to an error.

## Artworks (547)

| Annotated status (`gold_place_status`) | Artworks | Meaning |
|---|---|---|
| `depicted` | 291 | the artwork shows a real place, annotated with its coordinates |
| `none` | 199 | no place is shown (portrait, object, abstraction…) |
| `undeterminable` | 49 | cannot be decided from the metadata |
| `mentioned_only` | 8 | a place is mentioned but not shown |

Collections (`collection`): Art Institute of Chicago 447, Wikidata 71,
identified CSV records from Dresden 20 and Venice 9. Precision of the 291
places: exact place 157, neighbourhood 23, city 91, region 19, country 1.

**The gold standard is not a random sample of the collection**: a large
share of the AIC artworks was selected among titles that name a city
(`gold_sample_stratum` = `precise_diverse`) or among predictions to be
reviewed. The proportion of artworks depicting a place (53 %) is therefore
much higher than in the actual collection.

Annotations were made from the metadata **and** the image. A `none` artwork
may thus name a place in its title ("Portrait …, Paris"); an
`undeterminable` artwork may be decidable from the image: this is precisely
an interesting point of comparison.

## Predictions

Each artwork is predicted by a model **that did not see it during
training**: outer cross-validation with 5 folds (fixed partition, seed 0),
artworks sharing the same place being kept in the same fold. The column
`<model>_cv_fold` gives the test fold.

Two models:
- `v15`: the reference model (hierarchical logistic regression on features
  derived from the metadata: position of the place in the title, type of the
  geocoded place, NLI scores, etc.);
- `hybrid`: the same, plus features derived from a local LLM (Gemma3-4B) for
  the city, neighbourhood and exact-place levels.

For each model, two decisions:
- `<model>_all_levels_*`: the most specific level whose threshold is
  crossed, among all levels that have a threshold;
- `<model>_published_*`: the same decision restricted to the levels
  reliable enough to be published (mean precision ≥ 0.90 over 20
  partitions); **currently the country level only**.

Columns of a decision: `_level` (empty = abstention), `_place` (the name
read in the metadata), `_resolved_name`, `_latitude`, `_longitude`
(geocoded place), `_correct` (correct at the chosen level, according to the
rule above).

`<model>_evaluated` = false for `undeterminable` artworks (no decision can
be judged on them) and for 16 artworks used for training only
(`gold_sample_stratum` = `precise_keywords`).

## Reference results (this partition)

On the 482 evaluated artworks (of which 276 are `depicted`):

| Model, decision | Predictions | Correct | Precision | Recall | Predicted levels |
|---|---|---|---|---|---|
| v15, all levels | 36 | 31 | 0.861 | 0.112 | country 21, city 12, region 3 |
| v15, published | 27 | 25 | 0.926 | 0.090 | country 27 |
| hybrid, all levels | 43 | 37 | 0.860 | 0.133 | country 21, city 17, region 3, exact place 2 |
| hybrid, published | 27 | 25 | 0.926 | 0.090 | country 27 |

This partition is slightly below average: over 20 partitions, v15 "all
levels" reaches a mean precision of 0.865 ± 0.03 and a recall of 0.175; the
hybrid model 0.869 and 0.196. The model is deliberately conservative: it
abstains on the vast majority of artworks rather than risk an error.

## Comparing another method

For a fair comparison:
1. Use the same artworks (`<model>_evaluated` = true) and the same rule: an
   answer is correct if the annotated place is contained in the predicted
   place at the given level (exact place: within 0.25 km, or same name
   within 2 km; neighbourhood ≈ 2 km; city ≈ 25 km or same city; region;
   country).
2. Count a prediction on a `none` or `mentioned_only` artwork as an error,
   and an abstention as neither correct nor wrong (it lowers recall).
3. If the method is trained on these data, respect the folds of
   `<model>_cv_fold` (same place = same fold); otherwise the result will be
   overestimated.
4. The reference coordinates are `gold_latitude` / `gold_longitude`;
   `gold_country_code`, `gold_city` and `gold_region` come from reverse
   geocoding of these coordinates (Nominatim / OpenStreetMap).

## Licences and sources

AIC metadata and images: Art Institute of Chicago API (metadata CC0; images
according to `is_public_domain`). Wikidata: CC0. Dresden and Venice CSV
files: provenance to be checked before any public release. Reverse
geocoding: © OpenStreetMap contributors (ODbL). This package is intended
for research use within the project.
