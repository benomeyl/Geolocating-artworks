# Data
## Annotations géographiques

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

Le cache Nominatim est versionné afin que les anciennes réponses sans
hiérarchie d'adresse ne soient pas réutilisées.

Les données sont organisées en deux catégories :

- `raw/` : données originales récupérées depuis les APIs
- `processed/` : données nettoyées et transformées
