# Data
## Annotations géographiques

Les CSV `dresden_identified_data_ground_truth.csv` et
`venice_identified_data_ground_truth.csv` sont échantillonnés en conservant
les lignes qui ont une description (20 au maximum par ville), puis intégrés
au gold standard avec un aperçu non destructif et une écriture explicite :

```bash
python scripts/integrate_identified_csvs.py
python scripts/integrate_identified_csvs.py --write
```

Paris est volontairement exclu car ses lignes ne contiennent pas assez de
métadonnées. L'import conserve les identifiants d'image uniques, le `wd_id`
du lieu et la provenance. Les champs `city` et `region` sont définis pour
Dresden et Venise; aucune coordonnée n'est inventée si la source n'en fournit
pas.

Dans `raw/gold_standard/gold_dataset.json`, chaque `depicted_place` décrit
l'endroit représenté. `latitude` et `longitude` doivent localiser la cible
décrite par `name`, et non automatiquement le centre de la ville la plus
proche.

La valeur `precision` indique le niveau spatial auquel cette cible est
connue. Les valeurs canoniques sont `exact_place`, `neighborhood`, `city`,
`region`, `country` et `unknown`. `exact_place` couvre les bâtiments,
monuments, gares, rues et autres lieux précis. Les anciennes valeurs
`landmark` doivent être conservées pendant l'audit, puis converties après
vérification. Ne pas inférer la précision à partir du nom seul.

Les champs de hiérarchie recommandés dans `depicted_place` sont
`neighborhood`, `city`, `region`, `country` et `country_code`. Une valeur
inconnue doit rester `null` ou absente, jamais devinée. Garder
`geonameid`/`geonameid_match` et renseigner `coordinate_source` lorsque la
provenance des coordonnées est connue.

L'audit classe `city` comme non applicable pour les cibles de précision
`region` ou `country`, et `region` comme non applicable pour une cible
`country`. Ces valeurs ne sont pas comptées comme manquantes.

Pour auditer les champs manquants sans modifier les annotations :

```bash
python scripts/audit_geographic_annotations.py
```

Pour générer un export des étiquettes hiérarchiques des candidats, sans
entraîner le modèle :

```bash
python src/run_pipeline.py audit-labels \
	--records data/raw/gold_standard/gold_dataset.json \
	--output results/geographic_candidate_labels.json
```

La commande classe chaque candidat au niveau le plus précis justifié
(`exact_place`, `neighborhood`, `city`, `region`, `country`) dans
`label_level`. `label_status` vaut `match`, `negative` ou `unevaluable`; les
négatifs géographiques ont le niveau `no_match`. Nominatim est utilisé en
fallback; ajouter `--no-nominatim` pour rester hors ligne.

Pour entraîner le modèle hiérarchique et évaluer ses seuils sur des folds
externes groupés par cible géographique :

```bash
python src/run_pipeline.py train \
	--records data/raw/gold_standard/gold_dataset.json \
	--model-out results/model_hierarchical.joblib \
	--nominatim-user-agent "GeolocatingArtworks/0.1 (contact: votre-email)"
```

Les seuils visent une précision empirique minimale de 95 % (lieu exact),
93 % (quartier) et 90 % (ville/région/pays), avec un rappel œuvre souhaité
de 50 %. Un niveau insuffisamment représenté s'abstient. L'ancien modèle
par noms reste reproductible avec `train-legacy`.

Le cache Nominatim est versionné afin que les anciennes réponses sans
hiérarchie d'adresse ne soient pas réutilisées.

Les données sont organisées en deux catégories :

- `raw/` : données originales récupérées depuis les APIs
- `processed/` : données nettoyées et transformées
