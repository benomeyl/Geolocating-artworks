import sqlite3
import csv

conn = sqlite3.connect("geonames.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS geonames (
    geonameid INTEGER PRIMARY KEY,
    name TEXT,
    asciiname TEXT,
    alternatenames TEXT,
    latitude REAL,
    longitude REAL,
    feature_class TEXT,
    feature_code TEXT,
    country_code TEXT,
    population INTEGER
)
""")

colonnes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14]  # indices des colonnes utiles dans le fichier GeoNames

with open("data/gazetter/allCountries.txt", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    batch = []
    for row in reader:
        batch.append(tuple(row[i] for i in colonnes))
        if len(batch) >= 5000:
            cur.executemany(
                "INSERT OR IGNORE INTO geonames VALUES (?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            batch = []
    if batch:
        cur.executemany("INSERT OR IGNORE INTO geonames VALUES (?,?,?,?,?,?,?,?,?,?)", batch)

conn.commit()

# Index pour accélérer les recherches par nom
cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON geonames(name)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_asciiname ON geonames(asciiname)")
conn.commit()