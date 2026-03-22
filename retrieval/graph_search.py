"""
Knowledge Graph Search Module

Retrieves relationship context from the Neo4j knowledge graph
for a given user query. Used as the "graph" half of hybrid search.

The limit parameter defaults to config.yaml → retrieval.graph_limit.
"""

from graph_db.neo4j_store import get_driver, query_graph_for_query

# [Step 1] Import config loader for graph_limit default
from config.settings import get as cfg


def graph_search(query, driver=None, limit=None):
    """
    Retrieve context from the knowledge graph based on the query.

    Finds entities in the graph whose names match words in the query,
    then returns their relationships as context strings.

    Args:
        query: The user's question
        driver: Optional Neo4j driver (creates and closes one if not provided)
        limit: Max relationships to return per entity match
               (defaults to config.yaml → retrieval.graph_limit)

    Returns:
        List of relationship context strings (e.g., "Entity A --[relates_to]--> Entity B")
    """
    # [Step 1] Load default from config.yaml if not explicitly provided
    if limit is None:
        limit = cfg("retrieval", "graph_limit")

    # If no driver provided, create a temporary one and close it when done
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
