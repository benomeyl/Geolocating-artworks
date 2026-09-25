def get_location_info(name, conn):
    """
    Get location information from the GeoNames database based on the provided name.

    Args:
        name (str): The name of the location to search for.
        conn (sqlite3.Connection): The SQLite connection object.
    Returns:
        list: A list of tuples containing the location information.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT name, country_code, latitude, longitude, feature_code, population
        FROM geonames
        WHERE name = ? OR asciiname = ?
        ORDER BY population DESC
        LIMIT 5
    """, (name, name))
    return cur.fetchall()


def validate_location_candidate(loc_candidate, conn):
    """
    Validate the location candidate retrieved from the GeoNames database.

    Args:
        location_info (list): A list of tuples containing the location information.
        conn (sqlite3.Connection): The SQLite connection object.
    Returns:
        list: A list of validated location candidates with additional information.
    """
    results = get_location_info(loc_candidate, conn)

    if not results:
        return {"validate":False, "message": f"No location found for '{loc_candidate}'."}

    best = results[0]
    return {
        "validate": True,
        "name": best[0],
        "country_code": best[1],
        "latitude": best[2],
        "longitude": best[3],
        "feature_code": best[4]
        }

