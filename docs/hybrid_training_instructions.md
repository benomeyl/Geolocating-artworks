# Entraînement du modèle hybride LLM — instructions pour une session Claude

À donner tel quel à une nouvelle session Claude Code ouverte dans ce dépôt,
sur l'ordinateur le plus puissant (le calcul est sur processeur : plus il a
de cœurs, plus c'est rapide).

## Contexte

Le modèle hiérarchique (régression logistique, un seuil par niveau : lieu
exact, quartier, ville, région, pays) décide quel lieu publier pour chaque
œuvre. On a testé l'ajout de 5 features tirées d'un LLM local (Gemma3-4B,
`src/data/llm_places.py`, `LLM_FEATURE_COLUMNS`) :

- elles améliorent la ville (précision 0,759 → 0,849, deux fois plus de
  prédictions) et le lieu exact ;
- mais elles dégradent le pays (0,908 → 0,888), le seul niveau publié.

Le **modèle hybride** garde le modèle sans LLM pour le pays et la région et
utilise les features LLM pour la ville, le quartier et le lieu exact
(`train_hierarchical_model(extra_feature_columns=..., extra_feature_levels=...)`).
Il faut le mesurer contre la référence v15 sur les mêmes 20 découpages.

Tout ce qu'il faut est dans git : la table des candidats avec les features
LLM (`results/feature_table_v15_llm.pkl`), le gold
(`data/raw/gold_standard/gold_dataset.json`) et les options du modèle hybride
(`results/llm/hybrid_options.json`). Ni le dump AIC, ni GeoNames, ni Ollama,
ni réseau ne sont nécessaires.

## Ce qu'il faut faire

1. **Préparer l'environnement**
   - `git pull` sur la branche `top-down-geocoding`.
   - Python 3.13 de préférence, avec **les mêmes versions** que la machine
     de développement, pour que la référence se reproduise à l'identique :
     ```bash
     pip install "numpy==2.5.3" "pandas==3.0.5" "scikit-learn==1.9.1" "joblib==1.6.0" requests pytest
     ```
   - Vérifier : `python -m pytest -q` doit afficher 133 tests réussis.

2. **Lancer la comparaison** (depuis la racine du dépôt) :
   ```bash
   python scripts/compare_tables.py --jobs 8 --variant v15 results/feature_table_v15_llm.pkl --variant v15+llm-fin results/feature_table_v15_llm.pkl results/llm/hybrid_options.json --output results/compare_v15_hybrid.json > results/compare_v15_hybrid.log 2>&1
   ```
   - `--jobs` : nombre de découpages évalués en parallèle ; mettre le nombre
     de cœurs physiques de la machine (sans dépasser 20).
   - Durée estimée : 45 min pour `v15`, environ deux fois plus pour
     `v15+llm-fin` (deux modèles entraînés par découpage), soit 2 à 3 h sur
     8 cœurs, moins avec plus de cœurs. Empêcher la mise en veille.
   - La sortie n'apparaît dans le log qu'à la fin de chaque variante.

3. **Contrôler la reproduction de la référence** : dans le log, le bloc
   `=== v15 ===` doit donner précision 0,865 ± 0,029, faux positifs
   7,7 ± 2,5 et pays 0,908 déployé (chiffres de
   `results/compare_v15_llm.log`, calculés sur la machine de développement).
   Si ce n'est pas le cas, le noter dans le commit (versions des
   bibliothèques), sans rien corriger : la comparaison appariée du log reste
   valable car les deux variantes ont tourné sur la même machine.

4. **Rendre les résultats** : committer uniquement
   `results/compare_v15_hybrid.log` et `results/compare_v15_hybrid.json`,
   avec dans le message la durée et le nombre de cœurs, puis pousser la
   branche.

## À ne pas faire

- Ne pas modifier le code, les tables `.pkl`, le gold ni les options.
- Ne pas lancer `run_pipeline.py train` ni reconstruire les tables (il
  faudrait le dump AIC et les caches de la machine de développement).

## Ce que l'on regardera ensuite

Dans le bloc `--- v15+llm-fin − v15` : les faux positifs (doivent rester
stables), le pays (doit rester à 0,908, son modèle étant identique) et la
ville (attendue vers 0,85 avec plus de prédictions ; publiable seulement à
partir de 0,90).
