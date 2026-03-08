from graph_db.neo4j_store import get_driver, query_graph_for_query


def graph_search(query, driver=None, limit=15):
    """
    Retrieve context from the knowledge graph based on the query.

    Args:
        query: The user's question
        driver: Optional Neo4j driver (creates one if not provided)
        limit: Max relationships to return per entity match

    Returns:
        List of relationship context strings
    """
    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True

    try:
        results = query_graph_for_query(driver, query, limit=limit)
        return results
    finally:
        if close_driver:
            driver.close()
