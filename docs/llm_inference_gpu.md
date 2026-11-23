# Inférence LLM sur la machine GPU — instructions pour une session Claude

À donner tel quel à une nouvelle session Claude Code ouverte dans ce dépôt,
sur l'ordinateur équipé de la RTX 2070 Super.

## Contexte

Le projet géolocalise les œuvres de l'Art Institute of Chicago à partir de
leurs métadonnées. Un LLM local (via Ollama) lit chaque œuvre et renvoie, en
JSON, les lieux qu'elle nomme, leur nom usuel, leur ville et leur rôle
(représenté / origine / mentionné). Le code est dans `src/data/llm_places.py`.
Sur la machine de développement (CPU seul), Gemma3-4B prend environ 40 s par
œuvre : l'inférence est déportée sur cette machine GPU.

Les prompts sont **déjà construits** (ils dépendent d'un dump AIC absent de
git) : `results/llm/inputs_gold.jsonl` (547 œuvres du gold). Cette machine
n'a qu'à les envoyer au modèle et enregistrer les réponses. La vérification
des lieux (Wikidata, Nominatim) et l'analyse se feront ensuite sur la machine
de développement.

## Ce qu'il faut faire

1. **Préparer l'environnement**
   - `git pull` sur la branche `top-down-geocoding`.
   - Un environnement Python ≥ 3.10 avec `requests` suffit pour cette étape
     (`pip install -r requirements.txt`). Pas besoin de `transformers`, du
     dump AIC ni de GeoNames.
   - Installer Ollama (https://ollama.com/download), puis :
     `ollama pull gemma3:4b` et `ollama pull qwen3:4b`.
   - Vérifier que le modèle tourne **entièrement sur le GPU** : lancer une
     requête de test, puis `ollama ps` doit afficher `100% GPU`. Si ce n'est
     pas le cas (pilote NVIDIA, CUDA), le signaler à l'utilisateur avant de
     continuer : sur CPU, l'inférence prendrait des heures.

2. **Mesurer la vitesse sur 10 œuvres**
   ```bash
   python scripts/run_llm_places.py infer --model gemma3:4b --limit 10
   ```
   Attendu : environ 3 à 5 s par œuvre. Si c'est plus de 15 s, vérifier
   `ollama ps` (GPU) avant de lancer le reste.

3. **Gemma3-4B sur tout le gold (priorité)**
   ```bash
   python scripts/run_llm_places.py infer --model gemma3:4b
   ```
   Environ 30 à 45 min. Les réponses s'ajoutent ligne par ligne à
   `results/llm/answers_gemma3_4b.jsonl` ; en cas d'interruption, relancer la
   même commande reprend où elle s'est arrêtée.

4. **Qwen3-4B avec réflexion sur l'échantillon de 40 œuvres (si le temps le permet)**
   ```bash
   python scripts/run_llm_places.py infer --model qwen3:4b --think --ids results/llm/sample40_ids.txt
   ```
   Environ 30 à 60 s par œuvre, soit 20 à 40 min. Résultat dans
   `results/llm/answers_qwen3_4b_think.jsonl`. Si une œuvre dépasse le délai
   (30 min par défaut), elle est sautée et signalée par `[erreur]`.

5. **Rendre les résultats**
   - Committer **uniquement** `results/llm/answers_*.jsonl` (et, si utile, un
     court journal de la vitesse mesurée dans le message de commit), puis
     pousser la branche.
   - Ne pas committer `results/llm/infer_cache.sqlite` (ignoré par git).

## À ne pas faire

- Ne pas modifier le code, les prompts (`inputs_gold.jsonl`), le gold
  (`data/raw/gold_standard/`) ni les autres résultats : les réponses ne
  seraient plus comparables à celles déjà obtenues.
- Ne pas lancer d'entraînement ni `compare_local_llms.py` sur cette machine
  (ils demandent le dump AIC et les caches de la machine de développement).
- Ne pas changer de modèle ni de quantification (`gemma3:4b` et `qwen3:4b`
  tels que fournis par Ollama).

## Ensuite, sur la machine de développement

```bash
git pull
.venv/bin/python scripts/run_llm_places.py import
```
Les réponses entrent dans le cache local (`data/processed/llm_cache.sqlite`) ;
les analyses (comparaison, features du modèle) les relisent sans appeler de
modèle.
