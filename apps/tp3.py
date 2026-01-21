from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, count, when, abs as abs_spark, coalesce, to_date, regexp_replace, trim, initcap, \
    row_number, first, lit, lag, min, avg, ntile, percent_rank, countDistinct

print("=========== Initialisation ===========")
spark = SparkSession.builder \
    .appName("TP-analysis-of-athletic-performance-tp") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("/data/resultats_natation.csv", header=True, inferSchema=True)

df.show()

print("=========== Partie 1 : Nettoyage des données ===========")
print("1. Valeurs manquantes :")
# == Certains temps sont manquants (NULL ou 0)
df_cleaned = df.filter((col('temps_secondes').isNotNull()) & (col('temps_secondes') != 0))
df_cleaned.show()

# == Certaines catégories d'âge sont NULL
# Solution : Remplacer par 'categorie inconnue'
df_cleaned = df_cleaned.withColumn(
    "categorie_age",
    when(
        (col("categorie_age").isNull()),
        "categorie inconnue"
    ).otherwise(col("categorie_age"))
)
df_cleaned.show()
# == Certains noms d'athlètes sont vides
# Solution : Remplacer par 'nom inconnue'
df_cleaned = df_cleaned.withColumn(
    "nom",
    when(
        (col("nom").isNull()),
        "athlete inconnue"
    ).otherwise(col("nom"))
)
df_cleaned.show()

print("2. Doublons :")
# problem:
window_spec = Window.partitionBy("athlete_id", "epreuve", "nage", "date_competition")

df_duplicates = (
    df_cleaned.withColumn("cnt", count("*").over(window_spec))
      .filter(col("cnt") > 1)
      .drop("cnt")
)
df_duplicates.show()
# solution drop duplicate
df_cleaned = df_cleaned.dropDuplicates(
    ["athlete_id", "epreuve", "nage", "date_competition"]
)

df_cleaned.show()

print("3. Valeurs aberrantes :")

# == Temps négatifs (erreur de saisie)
#problem
df_cleaned.filter(col("temps_secondes") < 0).show()
#solution: remplace by abs value
df_cleaned = df_cleaned.withColumn(
    "temps_secondes",
    when(col("temps_secondes") < 0, abs_spark(col("temps_secondes"))).otherwise(col("temps_secondes"))
)
df_cleaned.show()

# == Temps > 300 secondes (disqualifications non marquées)
df_cleaned = df_cleaned.filter(col("temps_secondes") <= 300)
df_cleaned.show()

# == Âges < 10 ou > 80 (erreurs)
# df_cleaned.filter(10 > col("age") > 80)
#Problem
# df_cleaned.filter((col("age") < 10) | (col("age") > 80)).show(50)
#Solution: drop
df_cleaned = df_cleaned.filter((col("age") >=10) & (col("age") <= 80))
df_cleaned.show()

print("4. Formatage :")
# == Dates au format "YYYY-MM-DD" ou "DD/MM/YYYY" (inconsistant)
df_cleaned = df_cleaned.withColumn(
    "date_competition",
    coalesce(
        to_date("date_competition", "yyyy-MM-dd"),
        to_date("date_competition", "dd/MM/yyyy")
    )
)
df_cleaned.show()

# == Noms avec espaces en trop ou casse incohérente
df_cleaned = df_cleaned.withColumn(
    "nom",
    regexp_replace(trim(col("nom")), " +", " "))
df_cleaned.show()

# == Pays avec abréviations différentes (FR/FRA/France)
df_cleaned = (
    df_cleaned
    .withColumn(
        "pays",
        when(col("pays").isin("FR", "FRA"), "France")
        .otherwise(initcap(trim("pays")))
    )
)
df_cleaned.show()

print("=========== Partie 2 : Analyse avec Window Functions ===========")
print("=== Exercice 2.1 : Classement par épreuve et compétition ===")
w = Window.partitionBy("competition_id", "epreuve").orderBy("temps_secondes")

df_ranked_2_1 = (
    df_cleaned
    .withColumn("position", row_number().over(w))
    .withColumn("temps_premier", first("temps_secondes").over(w))
    .withColumn("ecart_avec_premier", col("temps_secondes") - col("temps_premier"))
    .withColumn("est_podium", when(col("position") <= 3, lit(True)).otherwise(lit(False)))
    .select(
        "athlete_id", "nom", "epreuve", "date_competition",
        "temps_secondes", "position", "ecart_avec_premier", "est_podium"
    )
)
df_ranked_2_1.show()

print("=== Exercice 2.2 : Progression personnelle ===")
# df = df.withColumn("ventes_mois_precedent", F.lag("ventes", 1).over(window_vendeur))\
#         .withColumn("ventes_mois_suivant", F.lead("ventes", 1).over(window_vendeur))\
#         .withColumn("variation_precedent", F.col("ventes") - F.lag("ventes", 1).over(window_vendeur))

w = Window.partitionBy("athlete_id", "epreuve").orderBy("date_competition")

#temps_precedent : temps de la compétition précédente
#amelioration_secondes : temps_precedent - temps_actuel (positif = amélioration)
#amelioration_pct : amélioration en pourcentage
#meilleur_temps_perso : record personnel sur cette épreuve jusqu'à cette date
#est_record_perso : booléen indiquant si c'est un nouveau record
df_ranked_2_2 = (
    df_cleaned
    .withColumn("temps_precedent", lag("temps_secondes", 1).over(w))
    .withColumn("amelioration_secondes", col("temps_precedent") - col("temps_secondes"))
    .withColumn("amelioration_pct", col("amelioration_secondes") * 100 / col("temps_precedent"))
    .withColumn("meilleur_temps_perso", min("temps_secondes").over(w))
    .withColumn("meilleur_temps_perso", min("temps_secondes").over(w))
    .withColumn("est_record_perso", col("meilleur_temps_perso") == col("temps_secondes"))
)
df_ranked_2_2.show()

print("=== Exercice 2.3 : Analyse par catégorie ===")
# Objectif : Comparer chaque performance à la moyenne de sa catégorie
#
# Attendu :
#
# temps_moyen_categorie : temps moyen de la catégorie sur cette épreuve (sur toute la saison)
# ecart_vs_moyenne : différence avec la moyenne (négatif = meilleur que la moyenne)
# percentile_categorie : dans quel quartile se situe l'athlète (1-4)
# rang_categorie : classement dans sa catégorie d'âge sur cette épreuve
# top_10_pct : booléen indiquant si dans les 10% meilleurs de sa catégorie
w_cat = Window.partitionBy("categorie_age", "epreuve")
w_rank = Window.partitionBy("categorie_age", "epreuve").orderBy("temps_secondes")

df_cat = (
    df_cleaned
    .withColumn("temps_moyen_categorie", avg("temps_secondes").over(w_cat))
    .withColumn("ecart_vs_moyenne", col("temps_secondes") - col("temps_moyen_categorie"))
    .withColumn("percentile_categorie", ntile(4).over(w_rank))
    .withColumn("rang_categorie", row_number().over(w_rank) )
    .withColumn("top_10_pct", percent_rank().over(w_rank) <= 0.10)
)
df_cat.show()

print("=== Exercice 2.4 : Polyvalence des nageurs ===")
# Objectif : Identifier les nageurs les plus polyvalents (performants sur plusieurs styles)
#
# Attendu :
#
# Par athlète, calculer :
#   nb_epreuves_differentes : nombre d'épreuves distinctes nagées
#   nb_styles_differents : nombre de styles de nage différents
#   meilleur_style : style où l'athlète a le meilleur classement moyen
#   rang_moyen_toutes_epreuves : rang moyen sur toutes ses participations
#   est_polyvalent : booléen (True si >= 3 styles différents)
# Filtrer : Athlètes ayant participé à au moins 5 compétitions

# Classement par épreuve
w_rank = Window.partitionBy("epreuve").orderBy("temps_secondes")

df_ranks = df_cleaned.withColumn("rang", row_number().over(w_rank))

df_poly = (
    df_ranks
    .groupBy("athlete_id", "nom")
    .agg(
        countDistinct("epreuve").alias("nb_epreuves_differentes"),
        countDistinct("nage").alias("nb_styles_differents"),
        avg("rang").alias("rang_moyen_toutes_epreuves"),
        countDistinct("competition_id").alias("nb_competitions")
    )
    .withColumn("est_polyvalent", col("nb_styles_differents") >= 3)
    .filter(col("nb_competitions") >= 5)
)

df_poly.show()