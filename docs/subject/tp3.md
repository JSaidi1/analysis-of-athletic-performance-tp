# Exercice PySpark : Analyse des Performances Sportives

## Contexte

Vous travailez pour une fédération sportive qui organise des compétitions de natation. Vous disposez des résultats de plusieurs athlètes sur différentes épreuves au cours de la saison 2024.

Votre mission est d'analyser les performances pour :

- Identifier les meilleurs nageurs par catégorie
- Analyser la progression individuelle
- Détecter les records personnels
- Comparer les performances aux moyennes de leur catégorie

## Description du Dataset

Le fichier `resultats_natation.csv` contient les colonnes suivantes :

| Colonne          | Type   | Description                                  |
| ---------------- | ------ | -------------------------------------------- |
| competition_id   | int    | Identifiant de la compétition                |
| athlete_id       | string | Identifiant de l'athlète                     |
| nom              | string | Nom de l'athlète                             |
| age              | int    | Âge de l'athlète                             |
| pays             | string | Pays d'origine                               |
| epreuve          | string | Type d'épreuve (50m, 100m, 200m)             |
| nage             | string | Style de nage (Libre, Dos, Brasse, Papillon) |
| date_competition | string | Date de la compétition                       |
| temps_secondes   | float  | Temps réalisé en secondes                    |
| categorie_age    | string | Catégorie d'âge (Junior, Senior, Master)     |

## Partie 1 : Nettoyage des données

Avant d'analyser les performances, vous devez nettoyer le dataset :

### Problèmes à corriger :

1. **Valeurs manquantes** :
   - Certains temps sont manquants (NULL ou 0)
   - Certains noms d'athlètes sont vides
   - Certaines catégories d'âge sont NULL

2. **Doublons** :
   - Certaines performances sont dupliquées (même athlète, même épreuve, même date)

3. **Valeurs aberrantes** :
   - Temps négatifs (erreur de saisie)
   - Temps > 300 secondes (disqualifications non marquées)
   - Âges < 10 ou > 80 (erreurs)

4. **Formatage** :
   - Dates au format "YYYY-MM-DD" ou "DD/MM/YYYY" (inconsistant)
   - Noms avec espaces en trop ou casse incohérente
   - Pays avec abréviations différentes (FR/FRA/France)

## Partie 2 : Analyse avec Window Functions

### Exercice 2.1 : Classement par épreuve et compétition

**Objectif** : Pour chaque compétition et chaque épreuve, classer les athlètes

**Attendu** :

- Colonnes : athlete_id, nom, epreuve, date_competition, temps_secondes
- `position` : classement dans l'épreuve (1er, 2ème, 3ème...)
- `ecart_avec_premier` : différence de temps avec le 1er (en secondes)
- `est_podium` : booléen indiquant si l'athlète est sur le podium (top 3)

**Filtrer** : Afficher uniquement le podium de chaque épreuve

---

### Exercice 2.2 : Progression personnelle

**Objectif** : Analyser l'évolution des performances de chaque athlète sur la saison

**Attendu** :

- Pour chaque athlète et chaque type d'épreuve (ex: "100m Libre")
- `temps_precedent` : temps de la compétition précédente
- `amelioration_secondes` : temps_precedent - temps_actuel (positif = amélioration)
- `amelioration_pct` : amélioration en pourcentage
- `meilleur_temps_perso` : record personnel sur cette épreuve jusqu'à cette date
- `est_record_perso` : booléen indiquant si c'est un nouveau record

**Trier** : Par athlète et date chronologique

---

### Exercice 2.3 : Analyse par catégorie

**Objectif** : Comparer chaque performance à la moyenne de sa catégorie

**Attendu** :

- `temps_moyen_categorie` : temps moyen de la catégorie sur cette épreuve (sur toute la saison)
- `ecart_vs_moyenne` : différence avec la moyenne (négatif = meilleur que la moyenne)
- `percentile_categorie` : dans quel quartile se situe l'athlète (1-4)
- `rang_categorie` : classement dans sa catégorie d'âge sur cette épreuve
- `top_10_pct` : booléen indiquant si dans les 10% meilleurs de sa catégorie

---

### Exercice 2.4 : Polyvalence des nageurs

**Objectif** : Identifier les nageurs les plus polyvalents (performants sur plusieurs styles)

**Attendu** :

- Par athlète, calculer :
  - `nb_epreuves_differentes` : nombre d'épreuves distinctes nagées
  - `nb_styles_differents` : nombre de styles de nage différents
  - `meilleur_style` : style où l'athlète a le meilleur classement moyen
  - `rang_moyen_toutes_epreuves` : rang moyen sur toutes ses participations
  - `est_polyvalent` : booléen (True si >= 3 styles différents)

**Filtrer** : Athlètes ayant participé à au moins 5 compétitions

---

### Exercice 2.5 : Tendance de performance

**Objectif** : Calculer une moyenne mobile pour détecter les tendances

**Attendu** :

- Pour chaque athlète sur son épreuve favorite (celle qu'il nage le plus)
- `moyenne_mobile_3comp` : moyenne des temps sur les 3 dernières compétitions
- `tendance` : "Amélioration", "Stable" ou "Dégradation"
  - Amélioration : temps actuel < moyenne mobile
  - Dégradation : temps actuel > moyenne mobile + 1 seconde
  - Stable : entre les deux
- `nb_competitions` : nombre de compétitions de l'athlète sur cette épreuve

---

### Exercice 2.6 : Performance relative par pays

**Objectif** : Comparer les athlètes à la performance moyenne de leur pays

**Attendu** :

- `meilleur_temps_pays` : meilleur temps du pays sur cette épreuve
- `temps_moyen_pays` : temps moyen du pays sur cette épreuve
- `rang_dans_pays` : classement de l'athlète parmi ses compatriotes
- `ecart_vs_meilleur_pays` : différence avec le meilleur de son pays
- `est_meilleur_pays` : booléen (meilleur temps de son pays sur cette épreuve)
