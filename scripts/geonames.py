"""Build the local GeoNames SQLite database from the allCountries.txt dump.

Download allCountries.zip from https://download.geonames.org/export/dump/,
unzip it into data/gazetter/, then run this script. The pipeline expects the
database at data/gazetter/geonames.db (move it there, or pass --output).
"""

import argparse
import csv
import sqlite3

# Indices of the useful columns of the GeoNames dump: id, name, ascii name,
# alternate names, latitude, longitude, feature class, feature code,
# country code and population.
COLUMNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14]
BATCH_SIZE = 5000
INSERT_ROW = "INSERT OR IGNORE INTO geonames VALUES (?,?,?,?,?,?,?,?,?,?)"


def build_database(input_path, output_path):
    """Load the tab-separated GeoNames dump into a `geonames` table indexed by name."""
    conn = sqlite3.connect(output_path)
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

    with open(input_path, encoding="utf-8") as f:
        batch = []
        for row in csv.reader(f, delimiter="\t"):
            batch.append(tuple(row[i] for i in COLUMNS))
            if len(batch) >= BATCH_SIZE:
                cur.executemany(INSERT_ROW, batch)
                batch = []
        if batch:
            cur.executemany(INSERT_ROW, batch)
    conn.commit()

    # Indexes used by gazetter.get_location_info.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON geonames(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_asciiname ON geonames(asciiname)")
    conn.commit()
    conn.close()


def main():
    """Parse the command line and build the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/gazetter/allCountries.txt")
    parser.add_argument("--output", default="geonames.db")
    args = parser.parse_args()
    build_database(args.input, args.output)


if __name__ == "__main__":
    main()
