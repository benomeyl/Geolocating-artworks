# Test sur 1 000 œuvres AIC : v15, hybride LLM et hybride exploratoire (27/11/2026)

## Question

Que publient les modèles quand on active tous leurs niveaux, sur des œuvres
AIC tirées au hasard hors gold ? On compare la référence v15 (sans LLM) et le
modèle hybride, qui ajoute les features du LLM Gemma3-4B pour le lieu exact,
le quartier et la ville.

## Protocole

- **Échantillon** : 1 000 œuvres du dump AIC (`data/raw/aic/aic_useful_fields.json`,
  139 752 œuvres), tirées avec la graine 42, hors gold. Les mêmes pour les
  trois modèles.
- **Commande** : `scripts/predict_sample_csv.py --size 1000 --seed 42 --use-active-levels`.
  Tous les niveaux qui ont un seuil publient, même ceux que la validation
  externe n'a pas validés (colonne `level_validated`).
- **Modèles** :

| Modèle | Fichier | Niveaux avec un seuil |
|---|---|---|
| v15 | `results/model_hierarchical_v15.joblib` (machine de développement) | ville, région, pays |
| hybride | `results/model_hierarchical_v15_hybrid.joblib` | ville, région, pays |
| hybride exploratoire | `results/model_hierarchical_v15_hybrid_explore.joblib` | lieu exact, ville, région, pays |

- **Entraînement des hybrides** : `scripts/train_from_table.py`, à partir de
  `results/feature_table_v15_llm.pkl`, sans reconstruire la table. Options
  `results/llm/hybrid_options.json`, et pour l'exploratoire
  `results/llm/hybrid_explore_options.json`. Logs dans
  `results/train_v15_hybrid*.log`.
  - Sur 20 découpages, l'hybride redonne exactement la comparaison
    `results/compare_v15_hybrid.log` : précision 0,869, 54,6 œuvres justes
    et 8,3 faux positifs par découpage.
  - Sur tout le gold, le modèle final de l'hybride ne trouve de seuil ni pour
    le lieu exact ni pour le quartier. La cible est 0,90, plus une marge de
    0,05 au réglage, avec seulement 62 groupes positifs.
- **Variante exploratoire** : cibles du lieu exact et du quartier abaissées
  à 0,80, pour donner un seuil au lieu exact.
  - Le lieu exact obtient un seuil (0,853) ; le quartier n'en obtient
    toujours pas.
  - En validation sur 20 découpages : lieu exact 0,73 (cumulée) avec
    5,7 prédictions par découpage ; ville 0,84 ; pays 0,91 ; faux positifs
    10,2 par découpage, contre 8,3 pour l'hybride.
  - Ses seuils de région et de pays changent aussi (0,987 et 0,984, contre
    0,978 et 0,982) : ce n'est pas un simple ajout de niveau.
- **Calcul** : Windows, Ryzen 7 9800X3D, RTX 2070 SUPER.
  - NER et NLI tournent sur le GPU (torch 2.14.1+cu130). Gemma3-4B passe par
    Ollama, sur le GPU, et ne lit que les œuvres qui ont des candidats.
  - Durées : v15 14,8 min ; hybride 21,9 min ; exploratoire 0,5 min, tout
    étant en cache.

## Résultats

| | v15 | hybride | hybride exploratoire |
|---|---|---|---|
| Sans candidat | 631 | 631 | 631 |
| Candidats sans décision | 317 | 316 | 310 |
| Œuvres avec une prédiction | 52 | 53 | 59 |
| pays | 42 | 42 | 45 |
| région | 7 | 7 | 6 |
| ville | 3 | 4 | 1 |
| lieu exact | – | – | 7 |

63 œuvres ont une prédiction dans au moins un modèle ; 39 ont la même dans
les trois.

### Différences entre les modèles

Notation : niveau publié : candidat géocodé (« – » : pas de prédiction).
Colonne LLM : ce que Gemma3 répond pour l'œuvre (lieu, rôle).

| Œuvre | v15 | hybride | exploratoire | LLM |
|---|---|---|---|---|
| 46433 Le Pont Neuf, Paris | pays : Paris | pays : Paris | **exact : Pont Neuf** | Pont Neuf, représenté |
| 139136 Album of Drawings of British Antiquities, including Stonehenge | – | **ville : Stonehenge** | **exact : Stonehenge** | Stonehenge, représenté |
| 114800 View from Eiffel Tower | ville : Tour Eiffel | ville : Tour Eiffel | exact : Tour Eiffel | Eiffel Tower, représenté (vue *depuis* la tour) |
| 47523 Chicago - State Street | pays : Chicago | pays : Chicago | **exact : State Street** | State Street, représenté |
| 86102 Ruins of Roman Forum, with Figures | – | – | **exact : Forum Romanum** | Roman Forum, représenté |
| 139433 The Pantheon, from the Terrace of the Luxembourg Gardens | ville : Panthéon | ville : Panthéon | **exact : Panthéon** | le place à Rome, mais le géocodage trouve celui de Paris |
| 46439 Saint Sulpice, Paris: La Petite Tour | – | – | **exact : Saint-Sulpice** | Saint Sulpice Church, représenté |
| 212442 Blacksher Garden Apartments, Mobile, Alabama | pays : Mobile | pays : Mobile | pays : Alabama | représenté |
| 236097 Michigan Avenue River Street Building, Chicago | région : Chicago | région : Chicago | pays : Chicago | représenté |
| 237555 Colvin House, Lake Forest, Illinois | pays : Illinois | pays : Illinois | pays : Lake Forest | représenté |
| 102271 United Church, Park Forest, Illinois, Floor Plan | pays : Illinois | pays : Illinois | pays : Park Forest | représenté |
| 75518 Unidentified Building, Chicago, Illinois, Elevation | pays : Chicago | pays : Chicago | pays : Illinois | mentionné |
| 74922 Robinson Children through the Doorway, Cabrini Green, Chicago | – | – | pays : Cabrini Row Houses | Cabrini Green, représenté |
| 210784 View down Black Cañon, from Mirror Bar | – | – | pays : Cañon School (lieu faux, pays sans doute juste) | Black Cañon, représenté |
| 6 photos *Protest Surrounding the Construction of Narita Airport* (208814, 208977, 208587, 208862, 208651, 208900) | – | – | pays : 成田国際空港 | Narita Airport, mentionné |
| 55455 Paul Arma's Hands, Paris | pays : Paris (erreur) | pays : Paris (erreur) | **– (erreur retirée)** | Paris, mentionné |
| 74895 The Belden Deli, Chicago | pays : Chicago | pays : Chicago | – (juste perdu) | représenté |
| 100696 Campanile, Piazza de San Marco, Venice | pays : Venice | pays : Venice | – (juste perdu) | représenté |
| 184512 Locust Grover Fruit Farm, Hudson River, New York | pays : Hudson River | pays : Hudson River | – | représenté |

### Prédictions identiques dans les trois modèles (39)

- **Pays (32) :**
  - Amsterdam, Holland ; Palais de Saint-Cloud ; Father and Child, 57th Street Beach, Chicago.
  - Mount Fuji Seen from the Senju Pleasure Quarter ; Little Houses, Chelsea ; Traghetto, Venice.
  - Bryan Lathrop House, Chicago ; Cornell Towers, Chicago ; Chicago Board of Trade Addition ; Mountains, Colorado.
  - Lowrise Apartment Building, Chicago ; Basilica of S. Croce in Gerusalemme ; *Pigeon, Paris* ; *Young Woman, Brooklyn*.
  - Yale School of the Fine Arts ; student festival « Rougevin », Paris ; St. Francis de Sales High School Students, Chicago.
  - Garden Glen Luxury Manor Homes, Schaumburg ; Auction Barn, Sleepyeye, Minnesota ; Vétheuil ; Belmont Avenue, Chicago.
  - Prentice Laboratory, Madison ; Lambert H. Ennis House (2 dessins) ; U.S. Route One, Baltimore ; Chicago (2).
  - View of rue Lacépède ; On Street Car, San Francisco ; Birmingham Holiday Inn ; Mixed Type Settlement, Berlin ; 34th Street, New York City.
- **Région (6) :**
  - Place de la Concorde, Paris ; In the Courtyard, Paris ; Clarendon Avenue School, Chicago.
  - Little Village Academy, Chicago ; Capitoline Hill ; Alexander Hamilton Memorial, Lincoln Park, Chicago.
- **Ville (1) :** Tomb of the Plautii on the Via Tiburtina.

## Constats

1. **L'hybride change à peine les prédictions sur des œuvres AIC aléatoires.**
   Il n'ajoute qu'une ville (Stonehenge). En validation sur le gold, il
   publiait deux fois plus de villes. Ici, le seuil de ville du modèle final
   est très haut (0,974), et les œuvres AIC aléatoires sont rarement des vues
   de ville.
2. **Les 7 lieux exacts de l'exploratoire paraissent tous justes.** Seule la
   Tour Eiffel est discutable : l'œuvre est une vue depuis la tour. C'est
   mieux que la précision de 0,73 mesurée en validation, mais sur 7 cas
   seulement.
3. **L'exploratoire déplace aussi des décisions de pays et de région.**
   - Il ajoute 6 photos de la série Narita (pays Japon, cas limite :
     manifestations sur le chantier) et Black Cañon (pays publié à partir
     d'un lieu faux).
   - Il retire une erreur claire (*Paul Arma's Hands*) mais aussi trois
     prédictions justes.
   - Il n'est donc pas comparable à l'hybride niveau par niveau.
4. **Les erreurs communes aux trois modèles restent les portraits et objets
   titrés « X, ville »** (*Pigeon, Paris*, *Young Woman, Brooklyn*).
   - Sur les 53 prédictions de l'hybride, Gemma3 répond « pas de lieu
     représenté » pour 16, dont ces erreurs.
   - Ce sont aussi plusieurs dessins d'architecture (« Floor Plan »,
     « Elevation »). Faut-il compter un plan comme représentant le bâtiment ?
     C'est une question d'annotation à trancher.

## Suite

- Relire les trois CSV pour mesurer la précision réelle par niveau. Colonnes
  `review_is_correct` / `review_notes`, liens vers l'image, réponse du LLM
  dans `llm_depicts_place` / `llm_places`.
- Les cas relus peuvent entrer dans le gold avec `scripts/integrate_manual_review.py`.

## Fichiers

- Prédictions : `results/test_sample_1000_v15.csv`,
  `results/test_sample_1000_v15_hybrid.csv`,
  `results/test_sample_1000_v15_hybrid_explore.csv`.
- Logs : `results/predict_1000_*.log`, `results/train_v15_hybrid*.log`.
- Code ajouté :
  - `scripts/train_from_table.py` ;
  - features LLM à la prédiction (`predict_with_hierarchical_model(llm=...)`,
    `model_uses_llm`) ;
  - `predict_sample_csv.py --llm-model` et colonnes `llm_*` ;
  - NER et NLI sur GPU quand CUDA est disponible.
