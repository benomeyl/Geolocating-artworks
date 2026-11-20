# Pipeline de géolocalisation des œuvres — état au 19/11/2026

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
   ├─ 5. un classifieur partagé par tous les niveaux .... train_hierarchical_model
   │       un seuil par niveau : exact_place, neighborhood, city, region, country
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
| `data/raw/gold_standard/gold_dataset.json` | Jeu annoté (440 œuvres) : Wikidata, CSV Dresde/Venise, revues manuelles AIC, lieux précis |
| `data/raw/aic/diverse_places_sample.json` | 582 œuvres AIC dont le titre nomme une ville (424 villes, 86 pays), recréé par `scripts/sample_diverse_places.py` |
| `data/raw/aic/aic_useful_fields.json` | 19 998 œuvres AIC, champs réduits (sert au test aléatoire) |
| `data/raw/artic-api-data/json/artworks/` | Dump AIC complet (134 078 fichiers, un par œuvre) |
| `data/gazetter/geonames.db` | GeoNames local (SQLite) |
| `data/processed/nominatim_cache.sqlite` | Cache des réponses Nominatim (évite de refaire les requêtes) |

Composition du gold (440 œuvres) : 195 `depicted` (115 lieux exacts,
16 quartiers, 46 villes, 17 régions, 1 pays), 190 `none`, 47
`undeterminable`, 8 `mentioned_only` ; 340 œuvres AIC et 100 d'autres
sources. Les œuvres AIC n'ont jamais été tirées au hasard : la part
d'œuvres AIC représentant un lieu (37 % dans le gold) n'est pas celle de la
collection.

Chaque annotation garde sa provenance (`ground_truth_source.sample_stratum`).
Les strates choisies par le modèle (`precise_keywords` : lignes annotées
dans l'ordre de `p_exact`) servent à l'entraînement mais jamais au test
(`TRAIN_ONLY_STRATA`). L'échantillon `precise_diverse`, exporté dans un
ordre aléatoire, sert aux deux.

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

Features calculées mais **hors du modèle** (testées sur 10 découpages
appariés, aucun gain au-delà du bruit, voir `results/compare_*.log`) :
`NLI_CONTRAST_COLUMNS` (contraste « montre » − « fait à »),
`HOMONYM_FEATURE_COLUMNS` (nombre d'homonymes, écart d'importance),
`MENTION_ROLE_COLUMNS` (mots juste avant la mention, position dans le
titre, nom d'institution), `PORTRAIT_FEATURE_COLUMNS` (nom de personne
dans le titre, sujets AIC de personnes, « X, ville » en fin de titre).

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

**Géocodage du plus large au plus précis** (`resolve_artwork_places`,
depuis le 05/11/2026) : les lieux d'une œuvre sont géocodés ensemble.

1. **Ancres** : pays, régions et lieux habités dont le nom correspond à la
   mention (ou lieu célèbre sous son nom local : « Florence » → « Firenze »),
   **n'importe où dans l'œuvre**. Sont exclus les noms génériques
   (`GENERIC_PLACE_WORDS` : « Park », « Town Hall ») et les mentions qui ne
   suivent qu'un indice d'origine (`ORIGIN_CUES` : « born in », « printed
   in »). Une ancre choisit l'alternative qui s'accorde avec les autres
   (même pays, proche) ; une ancre célèbre (importance ≥ 0,6) n'est révisée
   que par un pays ou une région (« Red House » ne déplace plus Berlin aux
   États-Unis). Un nom de pays n'est jamais révisé.
2. **Lieux précis** (les autres mentions) : la première alternative située
   dans la zone d'une ancre (ville : 30 km ; région : même pays et 500 km ;
   pays : même pays), l'ancre la plus proche dans le texte d'abord ; sinon
   une recherche Nominatim **limitée à la zone** (`search_within` :
   `countrycodes` + `viewbox` borné), puis « lieu, ancre » ; le résultat doit
   contenir tous les mots de la mention (« Cardiff Castle » ≠ « Cardiff
   Avenue »).
   **« X, Ville »** (depuis le 17/11/2026) : une ancre suivie directement
   d'une ancre plus importante dont elle est hors de la zone (« Notre Dame,
   Paris » → Notre Dame, Indiana ; « Saint Sulpice, Paris » → la commune de
   Saint-Sulpice) est traitée comme un lieu précis et cherchée dans cette
   ancre ; si elle n'y est pas trouvée, elle garde sa résolution d'ancre
   (« Takehara, Hiroshima »).
   **Wikidata** (option `--wikidata`, désactivée par défaut, 20/11/2026) :
   en dernier recours, un lieu précis introuvable dans la zone d'une ancre
   est cherché dans les labels et alias Wikidata (anglais, français,
   allemand, puis plein texte) ; seules les entités situées dans la zone,
   qui ne sont ni des localités ni des œuvres et dont un nom contient les
   mots distinctifs de la mention (abréviations et mots génériques
   traduits : « St. Ouen » → « Abbatiale Saint-Ouen ») sont gardées
   (`geolocation/wikidata.py`, `_wikidata_search`). Nominatim n'utilise
   Wikidata que pour classer ses résultats, pas pour chercher un nom.
3. **Échec** : si une ancre est juste à côté (≤ 80 caractères, « Le Nys,
   Amsterdam »), la mention n'a **pas de lieu** (sauf lieu célèbre, gardé et
   marqué `context_conflict`) : l'œuvre est placée par son ancre plutôt que
   par un homonyme. Un lieu célèbre situé dans un autre pays que l'ancre
   n'est gardé que s'il partage un mot distinctif avec la mention
   (« Clock Tower, Venice » n'est plus la Ghanta Ghar de Dehradun ;
   « Eiffel Tower » → « Tour Eiffel » reste accepté). Avec une ancre lointaine, le premier résultat est gardé
   et marqué `context_conflict`.
4. **Sans ancre** : premier résultat, sauf pour un nom générique (aucun
   lieu).

Chaque lieu porte `resolution` (`anchor`, `in_anchor`,
`searched_in_anchor`, `name_and_anchor`, `outside_anchors`, `global`).
Effet mesuré (v9 contre v7, mêmes découpages) : rappel +0,18 (10/10),
précision +0,03 (8/10).

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
| `exact_place` | candidat de niveau lieu exact à ≤ 0,25 km de la cible, ou portant le nom annoté à ≤ 2 km (rues, parcs, jetées) |
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

- **Étiquettes cumulatives** : au niveau « ville », est positif tout
  candidat juste au niveau ville ou plus précis.
- **Un modèle unique pour tous les niveaux** (`_fit_pooled_levels`, par
  défaut depuis le 18/11/2026) : les lignes des cinq niveaux sont empilées,
  chacune avec son niveau en feature (indicatrice du niveau + features
  multipliées par la « largeur » du niveau, de 0 pour le lieu exact à 1
  pour le pays). Les niveaux fins, qui ont peu d'exemples, profitent des
  poids appris sur les niveaux larges. `--separate-levels` rétablit une
  régression logistique par niveau.
- **Validation croisée groupée** : les œuvres qui partagent une même cible
  (ex. plusieurs vues de la Tour Eiffel) restent dans le même fold.
- **Seuil par niveau** : le plus permissif qui garde la précision
  hors-fold au-dessus de la cible **+ une marge** (`--threshold-margin`,
  0,05 ; cible 0,90 à tous les niveaux), avec au moins 5 prédictions et
  5 groupes. Sinon le niveau s'abstient (`precision_target_unmet`,
  `insufficient_positive_groups`…).
- **Probabilités hors-fold moyennées sur 5 découpages internes**
  (`--inner-repeats 5`) : la courbe de précision où l'on lit le seuil est
  moins irrégulière.
- **Prédiction par les modèles qui ont réglé le seuil**
  (`--prediction-model fold_average`, `FoldAverageClassifier`) : le score
  d'une nouvelle œuvre est la moyenne des modèles des folds internes. Un
  modèle réentraîné sur toutes les données (`refit`, ancien comportement)
  donnait des probabilités plus tranchées que celles du réglage du seuil et
  le franchissait trop facilement.
- **Validation externe** (`evaluate_hierarchical_grouped_cv`) : tout
  l'entraînement, seuils compris, est refait sur 5 folds externes ; les
  œuvres des strates `TRAIN_ONLY_STRATA` servent à l'entraînement de chaque
  fold mais ne sont jamais testées. Les métriques « OOF » affichées avant
  ont servi à régler les seuils et sont optimistes.
- **Validation répétée** (`evaluate_repeated_grouped_cv`, `--outer-repeats
  20`) : 20 découpages, moyenne ± écart-type. Les folds sont **stables**
  (`stable_group_folds`, depuis le 18/11/2026) : un groupe va dans le fold
  donné par un hash de la seed et de sa plus petite œuvre. Une correction
  du géocodage qui fusionne deux groupes ne déplace que les œuvres de ces
  groupes ; avant, `StratifiedGroupKFold` redistribuait tout et deux
  versions n'étaient plus comparables (fausse baisse de v12a).
  `scripts/compare_tables.py` compare deux tables ou réglages découpage par
  découpage, faux positifs d'abord.
- **Règle de déploiement D** (`deployable_levels`) : un niveau est activé
  si sa précision moyenne sur les découpages atteint la cible **et** si la
  borne basse de Wilson à 95 % de sa précision, sur toutes les prédictions
  des découpages réunies, atteint cible − 0,05. Ce second critère remplace
  depuis le 18/11/2026 « chaque découpage ≥ cible − 0,05 », qui devenait
  plus sévère à mesure qu'on ajoutait des découpages (avec 20, un seul
  découpage à 0,842 refusait un pays à 0,915 sur 1 272 prédictions). Avec
  `--outer-repeats 1`, l'ancienne règle (découpage fixe) s'applique.
- **Option non retenue** : `wilson_z` règle le seuil sur la borne basse de
  Wilson plutôt que sur la précision (voir §6).

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
| `python scripts/integrate_manual_review.py [--review-csv …] [--write]` | Ajoute un CSV annoté au gold (lit aussi les colonnes `review_*` des exports de lieux précis ; œuvres absentes lues dans le dump AIC) |
| `python scripts/sample_diverse_places.py --size 800` | Tire des œuvres AIC dont le titre nomme une ville (≤ 2 par ville, ≤ 30 par pays) |
| `python scripts/export_precise_places.py --input-csv … --order random --sample-stratum precise_diverse` | Exporte les œuvres ayant un candidat de niveau lieu exact, à annoter |
| `python scripts/relabel_city_in_title.py [--apply]` | Relecture des œuvres « X, ville » annotées sans lieu |
| `python scripts/integrate_identified_csvs.py [--write]` | Ajoute les CSV Dresde/Venise au gold |
| `python scripts/audit_geographic_annotations.py` | Liste les champs manquants des annotations |
| `python scripts/compare_feature_sets.py [--reuse-table] --output results/….json` | Compare des jeux de features sur la même table (validation externe, 10 découpages appariés, précision moyenne hors-fold par niveau) |
| `python scripts/compare_tables.py --variant NOM TABLE.pkl ['{options}'] … [--repeats 20]` | Compare des tables de features ou des options d'entraînement sur les mêmes découpages : faux positifs, précision par niveau, ce qui serait publié |
| `python -m pytest -q` | Tests (123) |

Lancer avec `.venv/bin/python` : l'interpréteur Anaconda de base n'a pas
`transformers`.

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

### Validation répétée et règle D (05/11/2026)

Moyenne ± écart-type sur 10 découpages ; colonnes Pays et Ville =
précision moyenne de ce niveau.

| Version | Changement | Œuvres testées | Précision | Rappel | Pays | Ville | Activé |
|---|---|---|---|---|---|---|---|
| v6 | + 16 lieux précis, ancienne règle | 345 | 0,825 ± 0,019 | 0,478 ± 0,029 | 0,854 | 0,796 | aucun |
| v7 | règle D | 345 | 0,840 ± 0,021 | 0,238 ± 0,028 | 0,904 | 0,670 | pays |
| v9 | géocodage du plus large au plus précis | 345 | 0,870 ± 0,021 | 0,419 ± 0,030 | 0,919 | 0,848 | pays |
| v10 | + 37 œuvres de l'échantillon diversifié | 377 | 0,820 ± 0,037 | 0,222 ± 0,020 | 0,902 | 0,797 | pays |

- v10 sur les seules 345 œuvres d'origine (mêmes découpages que v9) :
  précision 0,902, rappel 0,234, 4,1 fausses réponses par découpage contre
  9,9. Les œuvres diversifiées, plus difficiles, rendent les seuils plus
  stricts ; l'ancien gold surestimait la précision.
- Niveau lieu exact : ≈ 1,7 prédiction par découpage. Sur 115 lieux exacts
  annotés, 60 seulement ont un candidat juste au niveau exact (45 lieux
  distincts) ; les autres sont perdus au NER ou au géocodage.

### Stabilité des seuils et modèle unique (18/11/2026)

501 œuvres annotées (437 testées), géocodage v12. Comparaisons appariées
avec `scripts/compare_tables.py` sur les mêmes découpages ; « FP » =
fausses réponses par découpage.

| Étape | Précision | Rappel | FP | Effet apparié | Décision |
|---|---|---|---|---|---|
| v12 (référence, 10 découpages) | 0,832 | 0,194 | 9,2 | — | — |
| 1. prédiction par les modèles des folds | 0,844 | 0,163 | 7,1 | FP −2,1 (mieux 9/10) | gardée |
| 2. + 5 découpages internes | 0,853 | 0,186 | 7,6 | FP +0,5 (bruit), TP +5,4 | gardée |
| 3. seuil sur la borne de Wilson, z = 1,0 | 0,857 | 0,326 | 12,9 | FP +5,3 (pire 10/10) | rejetée |
| 3. seuil sur la borne de Wilson, z = 1,645 | 0,874 | 0,095 | 3,3 | TP −21,7, pays 0,856 | rejetée |
| 5. + modèle unique (20 découpages) | 0,890 | 0,242 | 7,1 | précision +0,035 (16/20), TP +14,4 (17/20), FP −0,2 | gardée |

- Avant, entre v11 et v12 (11 candidats différents sur 993), la précision
  variait de ± 0,05 d'un découpage à l'autre : l'instabilité venait du
  choix des seuils, pas des découpages.
- Un seuil plus strict ne retire pas les fausses réponses du pays
  (z = 1,645 : précision 0,856 au lieu de 0,900) : ce sont des erreurs
  sûres d'elles (portraits ou objets titrés « X, ville »), qu'il faudra
  traiter par de nouvelles informations, pas par le seuil.

**v13** (`model_hierarchical_v13.joblib`, `results/train_v13.log`) =
géocodage v12 + étapes 1, 2, 5 + nouvelle règle D, 20 découpages :

| Niveau | Précision moyenne | Toutes prédictions réunies | Prédictions par découpage | Activé |
|---|---|---|---|---|
| pays | 0,915 ± 0,029 | 0,915 (1 164 / 1 272), Wilson 0,898 – 0,929 | 63,6 | **oui** |
| ville | 0,898 ± 0,073 | 0,883 (234 / 265), Wilson 0,839 – 0,916 | 13,2 | non (cible 0,90) |
| région | 0,840 ± 0,080 | 0,828 | 13,4 | non |
| lieu exact / quartier | — | — | < 1 | non |

Toutes prédictions confondues : précision 0,890 ± 0,027, rappel
0,242 ± 0,032 (v11 : 0,858 / 0,197).

### Wikidata en dernier recours (20/11/2026)

Sur les 64 œuvres dont le lieu exact annoté n'arrive pas comme candidat
exact, Wikidata contient la bonne entité pour 39. La version prudente
(`--wikidata`, table `feature_table_v14_wikidata.pkl`) résout 27
candidats, dont 5 nouveaux lieux exacts justes ; comparée à v13 sur les
mêmes 20 découpages : FP +0,35 (bruit), TP +1,2, pays 0,915 → 0,916,
ville 0,898 → 0,873. Non adoptée. La plupart des monuments perdus
n'atteignent pas Wikidata : pas d'ancre, ville dans le nom du monument
(« Cathedral of Mainz »), monument non extrait par le NER, ancre fausse.

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

- **Lieux exacts** : ≈ 70 candidats justes pour 45 lieux distincts ; 55 des
  115 lieux exacts annotés n'arrivent jamais comme candidat exact (ville
  seule 33, région 9, pays 5, ailleurs 4, non géocodé 3, non extrait 1).
- **Métadonnées seules** : un portrait ou une photo abstraite titrés
  « X, ville » ne peuvent pas être distingués d'une vue de la ville ; les
  sujets AIC ne sont renseignés que pour 16 % des œuvres figuratives.
- **67 % des œuvres AIC tirées au hasard n'ont aucun candidat** : surtout
  des titres sans lieu (« Untitled », « Vase ») ; 7 % seulement ont une
  description.
- **Calcul** : NLI sur CPU, ≈ 3 h 15 pour prédire 582 œuvres ; Nominatim
  limité à une requête par seconde.
- **Descriptions longues** tronquées à 512 tokens par le NER.
- **Entraînement plus lent** depuis le modèle unique : ≈ 60 s par
  entraînement, 45 min à 1 h pour `train` avec 20 découpages (8 cœurs).
- **Wikidata et Nominatim** : Nominatim n'utilise Wikidata/Wikipédia que
  pour classer les résultats (importance), pas pour trouver un lieu par
  son nom ; les noms cherchables sont ceux d'OpenStreetMap. Les noms
  anglais descriptifs absents d'OSM (« St. Ouen, Rouen ») sont donc perdus :
  une recherche dans les labels et alias Wikidata est la piste suivante.
- Pistes et tâches détaillées dans `docs/feuille_de_route_2026-11-13.html` :
  finir l'annotation de l'échantillon diversifié (51 œuvres restantes),
  récupérer les lieux exacts perdus, dictionnaire de monuments, labels
  Wikidata, comparaison avec un LLM, accélération du NLI.
