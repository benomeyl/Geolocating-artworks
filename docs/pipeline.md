# Pipeline de géolocalisation des œuvres — état au 01/11/2026

## 1. Objectif

Pour une œuvre (aujourd'hui surtout de l'Art Institute of Chicago), trouver
le **lieu représenté** à partir de ses seules **métadonnées textuelles**
(titre, description, sujets…), sans regarder l'image. Si le lieu exact est
incertain, on préfère répondre à un niveau plus général (ville, région,
pays) ou s'abstenir plutôt que de donner un faux positif.

Ce qu'on ne veut **pas** détecter : le lieu de création, de naissance de
l'artiste, de provenance ou d'exposition.

## 2. Vue d'ensemble

```
 œuvre (JSON)
   │
   ├─ 1. enrichissement depuis le dump AIC local ........ src/api/aic.py
   │
   ├─ 2. NER : lieux mentionnés (LOC) + nettoyage ....... filtering_logreg.extract_candidate_artwork
   │       un « candidat » = un nom de lieu dans un champ
   │
   ├─ 3. regroupement par nom + features textuelles ..... candidate_features, artwork_context_features
   │
   ├─ 4. géocodage : GeoNames puis Nominatim ........... src/geolocation/gazetter.py
   │       → coordonnées, hiérarchie (ville/région/pays), niveau du candidat
   │
   ├─ 5. un classifieur par niveau géographique ......... train_hierarchical_model
   │       exact_place, neighborhood, city, region, country
   │
   └─ 6. décision : le niveau le plus précis dont le seuil est franchi
           et qui est « déployable », sinon abstention . predict_with_hierarchical_model
```

Tout le code actif du modèle est dans `src/data/filtering_logreg.py`.
`src/data/filtering.py` est l'ancienne version (score heuristique à poids
manuels) : plus aucun code ne l'importe.

## 3. Données

| Fichier | Contenu |
|---|---|
| `data/raw/gold_standard/gold_dataset.json` | Jeu annoté (387 œuvres) : Wikidata, CSV Dresde/Venise, revues manuelles AIC |
| `data/raw/aic/aic_useful_fields.json` | 19 998 œuvres AIC, champs réduits (sert au test aléatoire) |
| `data/raw/artic-api-data/json/artworks/` | Dump AIC complet (134 078 fichiers, un par œuvre) |
| `data/gazetter/geonames.db` | GeoNames local (SQLite) |
| `data/processed/nominatim_cache.sqlite` | Cache des réponses Nominatim (évite de refaire les requêtes) |

Composition du gold : les œuvres AIC représentent un lieu dans **21 %** des
cas (29/136), les autres sources dans **70 %** des cas. Ce déséquilibre
explique une partie des difficultés (voir §8).

Chaque annotation contient `place_status` (`depicted`, `none`,
`mentioned_only`, `undeterminable`) et, si `depicted`, un `depicted_place`
avec nom, coordonnées, ville/région/pays et `precision` (le niveau auquel
la cible est connue). Détails : `data/README.md`.

## 4. Les étapes en détail

### 4.1 Enrichissement AIC (`enrich_records_from_dump`)

Complète les œuvres AIC (identifiant numérique) avec des champs du dump :

| Champ | Usage |
|---|---|
| `alt_titles` | source de candidats, comme le titre |
| `artwork_type_title` | feature `object_non_pictorial` (vase, monnaie, sculpture…) |
| `place_of_origin` | feature `in_place_of_origin` |
| `artist_display` | feature `in_artist_display` (biographie : « born Leiden ») |
| `exhibition_history`, `provenance_text`, `credit_line` | feature `in_provenance_or_exhibition` |
| `latitude`, `longitude`, `gallery_title` | **jamais utilisés** : c'est la salle du musée à Chicago |

Seul `alt_titles` produit des candidats ; les autres champs ne servent
qu'à qualifier un candidat déjà trouvé ailleurs.

### 4.2 Extraction NER (`extract_candidate_artwork`)

- Modèle : `Babelscape/wikineural-multilingual-ner`, agrégation `first`.
- Champs lus : titre, `alt_titles`, description, description courte,
  thèmes, sujets, styles.
- Seules les entités `LOC` sont gardées, à partir d'un score NER de 0,2 (le
  score lui-même devient une feature).
- `_clean_entity_spans` répare les sorties du NER : le nom est relu dans le
  texte source, les mots coupés sont complétés, les morceaux collés par
  `.`, `-` ou `'` fusionnés (`N.Y`), `of Control` → `Line of Control`,
  `d'Aix-en-Provence` → `Aix-en-Provence` ; les mots-outils et lettres
  isolées sont écartés.

### 4.3 Features d'un candidat

Les occurrences d'un même nom dans une œuvre sont regroupées en un
candidat. Ses features :

- **Textuelles** : champs où il apparaît (`field_title`, …), nombre de
  champs, marqueur positif proche (« view of », « vue de »…), marqueur
  négatif proche (« born in », « exhibited in »…), score NER max.
- **Contexte AIC** : `object_non_pictorial`, `in_place_of_origin`,
  `in_artist_display`, `in_provenance_or_exhibition`.
- **Géographiques** : niveau du lieu géocodé (`geo_level_*`), source
  (GeoNames/Nominatim), hiérarchie disponible, importance et rang
  Nominatim, ressemblance entre le candidat et le nom trouvé
  (`geo_name_similarity`) et conflit avec le contexte
  (`geo_context_conflict`).
- **NLI** (`src/data/nli.py`, depuis la v5) : un modèle d'inférence
  multilingue zero-shot (`mDeBERTa-v3-base-mnli-xnli`) lit le titre et le
  passage autour de la mention, et donne la probabilité de 5 hypothèses :
  « shows a view of the place X » (`nli_view`), « the scene … is located
  in X » (`nli_scene`), « X is a geographic location » (`nli_is_place`),
  « was made in X » (`nli_made_in`), « the artist was born or lived in X »
  (`nli_artist_life`). « This artwork depicts X » est évité : c'est vrai
  d'un vase pour « Vase ». Scores en cache dans
  `data/processed/nli_cache.sqlite` (≈ 0,4 s par paire sur CPU).

### 4.4 Géocodage (`gazetter.py`)

**Alternatives d'un nom** (`location_alternatives`) :

1. GeoNames local, par nom exact, plus les pays par leurs noms alternatifs
   (GeoNames nomme « Kingdom of Spain » ce que les titres appellent
   « Spain »). Une ville (`PPL*`) proche d'une unité administrative homonyme
   passe devant (Paris `PPLC` avant Paris `ADM3`).
2. Seuls les pays, unités administratives et lieux habités d'au moins
   1 000 habitants sont « fiables » (pas l'hôtel « Louvre » en Grèce ni le
   hameau « Puerta del Sol » au Mexique).
3. Nominatim est toujours interrogé. Ses résultats non représentables sont
   écartés : commerces, restaurants, bouches de métro, simples bâtiments,
   chemins, œuvres d'art de rue (« Microcosm » → une boutique de
   photocopies en Biélorussie).
4. Les entrées GeoNames fiables et les résultats Nominatim sont **classés
   ensemble par importance Nominatim** (liée à Wikipédia) : une entrée
   GeoNames prend l'importance du résultat Nominatim qui décrit le même lieu
   (même pays, à moins de 25 km, 100 km pour une unité administrative), ou
   0,25 si Nominatim ne la connaît pas. Ainsi le lieu célèbre l'emporte sur
   ses homonymes : Patmos l'île grecque (0,54) et non Patmos dans l'Arkansas
   (0,36) ; la Normandie (0,72) et non Normandy (Missouri) ; le Corcovado à
   Rio (0,45) et non un bourg de 1 500 habitants à Porto Rico. Un résultat
   Nominatim confirmé par une entrée GeoNames homonyme à moins de 5 km
   reçoit un petit bonus (Point Lobos en Californie plutôt qu'une rue du
   Texas).
5. Les entrées GeoNames non fiables ne servent que si rien d'autre n'est trouvé.

**Désambiguïsation par le contexte** (`resolve_artwork_places`) : les
lieux d'une œuvre sont géocodés ensemble. Un lieu cité à moins de 80
caractères d'un « point d'ancrage » du même champ (pays, région ou lieu
habité dont le nom correspond bien) choisit l'alternative du même pays ou à
moins de 150 km (Logan County dans l'Illinois à côté de Cornland). Sans
alternative compatible, on cherche « lieu, contexte » dans Nominatim ; en
cas d'échec, le premier résultat est gardé et marqué `context_conflict`. Un
nom de pays n'est jamais remis en cause par un lieu plus petit, ni un lieu
très connu (importance ≥ 0,6 : Rome, Paris) par la recherche « lieu,
contexte » (« Rome, France » trouvait la station de métro Rome à Paris) :
il est gardé et marqué `context_conflict`.

**Hiérarchie** ville/région/pays : par géocodage inverse Nominatim.

Seuls les 3 candidats au meilleur score NER de chaque œuvre sont géocodés à
la prédiction (`--geocode-top-k`).

**Niveau d'un lieu** (`_candidate_geographic_level`) :

- GeoNames : `PCL*` → pays ; `ADM1`, `ADM2`, `RGN` → région ;
  `PPL*`, `ADM3+` → ville ; `PPLX` → quartier ; autre → lieu exact.
- Nominatim : `addresstype`/type (`city`, `suburb`, `state`…), sinon
  `place_rank` pour les limites administratives (≤ 4 pays, ≤ 12 région,
  ≤ 19 ville, ≤ 24 quartier, ≥ 25 lieu exact).

### 4.5 Étiquettes d'entraînement (`geographic_candidate_label`)

Pour chaque candidat d'une œuvre annotée, on compare son lieu géocodé à la
cible annotée et on note **le niveau le plus précis qui correspond** :

| Niveau | Condition principale |
|---|---|
| `exact_place` | candidat de niveau lieu exact à ≤ 0,25 km de la cible |
| `neighborhood` | même quartier, ou lieu exact à ≤ 2 km |
| `city` | même ville, ou à ≤ 25 km |
| `region` / `country` | même région / même pays |

- Les pays sont comparés par code ISO, les régions et villes aussi avec la
  hiérarchie obtenue par géocodage inverse des coordonnées de la cible
  (ce qui règle `Saxony` / `Sachsen`).
- Œuvre `none` ou `mentioned_only` : tous ses candidats sont négatifs.
- Candidat non géocodé ou cible sans coordonnées : `unevaluable`, exclu de
  l'entraînement (jamais forcé à négatif).

### 4.6 Entraînement (`train_hierarchical_model`)

- **Une régression logistique par niveau**, avec des étiquettes
  cumulatives : le modèle « ville » compte comme positif tout candidat
  juste au niveau ville ou plus précis.
- **Validation croisée groupée** : les œuvres qui partagent une même cible
  (ex. plusieurs vues de la Tour Eiffel) restent dans le même fold, pour
  ne pas surestimer les performances.
- **Seuil par niveau** : le plus permissif qui garde la précision
  hors-fold au-dessus de la cible (95 % lieu exact, 93 % quartier, 90 %
  ville/région/pays), avec au moins 5 prédictions et 5 groupes. Sinon le
  niveau s'abstient (`precision_target_unmet`,
  `insufficient_positive_groups`…).
- **Validation externe** (`evaluate_hierarchical_grouped_cv`) : on refait
  tout l'entraînement, seuils compris, sur 5 folds externes. Un niveau est
  **déployable** seulement si sa précision externe atteint aussi la cible.
  C'est la seule métrique honnête ; les métriques « OOF » affichées avant
  ont servi à choisir les seuils et sont optimistes.

### 4.7 Prédiction (`predict_with_hierarchical_model`)

Pour chaque candidat, chaque niveau déployable compatible donne un score.
On retient le niveau **le plus précis** dont le seuil est franchi ; entre
candidats, le plus précis puis le meilleur score. La « confiance » affichée
est la précision externe mesurée pour ce niveau. Sans aucun niveau franchi,
l'œuvre est `uncertain`.

## 5. Commandes

| Commande | Rôle |
|---|---|
| `python src/run_pipeline.py train --records data/raw/gold_standard/gold_dataset.json --model-out results/…joblib` | Entraîne et évalue (à lancer depuis `src/` ou adapter les chemins) |
| `python src/run_pipeline.py audit-labels --records …` | Exporte les étiquettes des candidats pour inspection, sans entraîner |
| `python src/run_pipeline.py predict --records … --model-in … --output …json` | Prédit sur de nouvelles œuvres |
| `python scripts/predict_sample_csv.py --model … --output results/….csv` | Test sur 500 œuvres AIC aléatoires hors gold, export CSV avec liens image |
| `python scripts/create_manual_review_sample.py …` | Échantillon stratifié de prédictions à annoter |
| `python scripts/create_annotation_csv.py` | CSV simple à annoter à partir du test aléatoire (œuvres `uncertain`) |
| `python scripts/integrate_manual_review.py [--review-csv …] [--write]` | Ajoute un CSV annoté au gold |
| `python scripts/integrate_identified_csvs.py [--write]` | Ajoute les CSV Dresde/Venise au gold |
| `python scripts/audit_geographic_annotations.py` | Liste les champs manquants des annotations |
| `python scripts/compare_feature_sets.py [--reuse-table] --output results/….json` | Compare des jeux de features sur la même table (validation externe + précision moyenne hors-fold par niveau) |
| `python -m pytest -q` | Tests (84) |

## 6. Résultats

Mêmes 253 œuvres annotées ; « externe » = validation externe groupée
(§4.6), « test AIC » = 500 œuvres AIC tirées au hasard hors gold
(`scripts/predict_sample_csv.py`, seed 42).

| Modèle | Différence | Précision ext. ville / région / pays | Niveaux déployés | Prédictions test AIC |
|---|---|---|---|---|
| v1 `model_hierarchical.joblib` | avec `has_context_metadata` | 0,77 / 0,86 / 0,90 | pays | 3 / 500 (justes) |
| v2 `model_hierarchical_v2.joblib` | sans `has_context_metadata` | 1,0 / 1,0 / 1,0 sur 3–4 groupes seulement | aucun | 0 |
| v3 `model_hierarchical_v3.joblib` | v2 + `object_non_pictorial` | 0,81 / 0,875 / 0,889 | aucun | 0 |

- `has_context_metadata` (1 pour toute œuvre AIC) servait de raccourci de
  source : poids ≈ −3,4, il pénalisait toute œuvre AIC. Il n'est plus une
  feature (il reste calculé pour que la v1 fonctionne).
- `object_non_pictorial` reçoit le poids attendu (−0,75 à −2,1) et
  rapproche les niveaux ville/région/pays de la cible de 90 %, sans
  l'atteindre.
- Sur le test AIC : 336 œuvres sans candidat, 164 `uncertain`. Les cas
  juste sous le seuil sont presque tous corrects (Navy Pier, Chicago ;
  The Hippodrome, London ; St. Ouen, Rouen).
- Les niveaux lieu exact et quartier n'atteignent jamais la cible : 20 à
  23 groupes positifs ne suffisent pas.

### Sur le gold de 387 œuvres (01/11/2026)

Comparaisons faites avec `scripts/compare_feature_sets.py` (même table,
mêmes folds). « Préc. moyenne » = précision moyenne hors-fold sans seuil
(aire sous la courbe précision-rappel), moyenne sur 5 découpages : plus
stable que la validation externe, qui repose sur quelques dizaines de
prédictions.

| Version | Précision / rappel ext. | Préc. moyenne ville / région / pays | Rappel pays à 90 % de précision |
|---|---|---|---|
| v4 (`model_hierarchical_v4.joblib`) | 0,773 / 0,215 | 0,681 / 0,711 / 0,766 | 0,09 |
| v4 + patrons de titre | 0,600 / 0,076 | 0,667 / 0,700 / 0,767 | — (abandonné) |
| v4 + NLI | 0,739 / 0,108 | 0,704 / 0,740 / 0,802 | 0,09 |
| v4 + nouveau géocodage | 0,737 / 0,266 | 0,732 / 0,764 / 0,787 | 0,04 |
| **v5** = v4 + NLI + nouveau géocodage (`model_hierarchical_v5.joblib`) | **0,816 / 0,392** | **0,768 / 0,797 / 0,834** | **0,27** |

- Les patrons de titre (« View of X », « X, Y », « Vase from X »…) étaient
  trop rares et redondants : la plupart des erreurs viennent des
  descriptions et de noms d'objets lus comme des lieux (« Vase », « Cane »).
- La v5 atteint 0,90 de précision OOF pour ville, région et pays, mais
  seulement 0,82–0,83 en validation externe : aucun niveau déployé.
- Erreurs restantes en tête de classement : rôle du lieu discutable
  (« Perv's House, Chicago », « Jewelled Bowl in the Treasury of San Marco,
  Venice ») et mentions mal découpées par le NER (« Paris Street » pour
  *Paris Street; Rainy Day*, « South Gate of »).

## 7. Fichiers de code

| Fichier | Rôle |
|---|---|
| `src/data/filtering_logreg.py` | Candidats, features, étiquettes, entraînement, prédiction, évaluation |
| `src/data/nli.py` | Scores NLI zero-shot avec cache (features `nli_*`) |
| `src/geolocation/gazetter.py` | GeoNames, client Nominatim avec cache, distance |
| `src/api/aic.py` | API et dump AIC, enrichissement |
| `src/run_pipeline.py` | Ligne de commande `train` / `audit-labels` / `predict` / `train-legacy` |
| `src/data/ner.py`, `src/main.py` | Anciens points d'entrée (modèle par noms) |
| `src/data/filtering.py` | Ancien score heuristique, plus utilisé |

## 8. Limites connues et pistes

- **Peu d'exemples AIC positifs** (29) : c'est le principal frein. Les
  niveaux lieu exact et quartier n'ont que 20 à 23 groupes positifs.
  Priorité : annoter les œuvres `uncertain` du test aléatoire.
- **67 % des œuvres AIC tirées au hasard n'ont aucun candidat** : surtout
  des titres sans lieu (« Untitled », « Vase ») ; 7 % seulement ont une
  description.
- **Géocodage** : un lieu cité seul reste ambigu (« Conway Castle » →
  Floride ; aucune source ne connaît cette orthographe de Conwy) ; un lieu
  sans ancrage fiable à côté n'est pas désambiguïsé.
- **Descriptions longues** tronquées à 512 tokens par le NER.
- Pistes : liaison Wikidata, désambiguïsation par le contexte, LLM local
  comme feature, second NER (GLiNER multilingue).
