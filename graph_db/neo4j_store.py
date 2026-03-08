import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def get_driver():
    """Create and return a Neo4j driver instance."""
    uri = os.getenv("NEO4J_URI", "neo4j+ssc://d23c6a98.databases.neo4j.io")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    # Use neo4j+ssc:// to skip full certificate chain verification (needed for Aura on some platforms)
    if uri.startswith("neo4j+s://"):
        uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1)
    return GraphDatabase.driver(uri, auth=(user, password))


def clear_graph(driver):
    """Delete all nodes and relationships in the graph."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def store_entities(driver, entities):
    """Store extracted entities as nodes in Neo4j."""
    with driver.session() as session:
        for entity in entities:
            name = entity["name"]
            entity_type = entity.get("type", "Entity")
            source_chunk = entity.get("source_chunk", -1)
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.source_chunk = $source_chunk
                """,
                name=name, type=entity_type, source_chunk=source_chunk
            )


def store_relationships(driver, relationships):
    """Store extracted relationships as edges in Neo4j."""
    with driver.session() as session:
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            relation = rel["relation"]
            source_chunk = rel.get("source_chunk", -1)
            session.run(
                """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:RELATES_TO {relation: $relation}]->(b)
                SET r.source_chunk = $source_chunk
                """,
                source=source, target=target, relation=relation, source_chunk=source_chunk
            )


def build_knowledge_graph(driver, entities, relationships):
    """Build the full knowledge graph: clear existing data, store entities and relationships."""
    clear_graph(driver)
    store_entities(driver, entities)
    store_relationships(driver, relationships)
    print(f"Knowledge graph built: {len(entities)} entities, {len(relationships)} relationships.")


def query_graph(driver, entity_name, max_depth=2, limit=10):
    """
    Query the knowledge graph for entities and relationships related to a given entity name.
    Returns a list of readable relationship strings.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            CALL {
                WITH e
                MATCH path = (e)-[r:RELATES_TO*1..""" + str(max_depth) + """]->(related)
                RETURN e AS source, r AS rels, related, nodes(path) AS path_nodes
                LIMIT $limit
            }
            RETURN source.name AS source_name, source.type AS source_type,
                   related.name AS related_name, related.type AS related_type,
                   [rel IN rels | rel.relation] AS relations
            """,
            name=entity_name, limit=limit
        )

        graph_context = []
        for record in result:
            source = record["source_name"]
            target = record["related_name"]
            relations = record["relations"]
            rel_chain = " -> ".join(relations)
            graph_context.append(f"{source} --[{rel_chain}]--> {target}")

        return graph_context


def query_graph_for_query(driver, query, limit=15):
    """
    Search the graph for any entities whose names appear in the query.
    Returns relationship context strings.
    """
    # Extract potential entity mentions by querying all entity names
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        all_entity_names = [record["name"] for record in result]

    # Find entities mentioned in the query
    query_lower = query.lower()
    matching_entities = [
        name for name in all_entity_names
        if name.lower() in query_lower or any(
            word in query_lower for word in name.lower().split() if len(word) > 3
        )
    ]

    # Query graph for each matching entity
    all_context = []
    seen = set()
    for entity_name in matching_entities:
        results = query_graph(driver, entity_name, max_depth=2, limit=limit)
        for ctx in results:
            if ctx not in seen:
                seen.add(ctx)
                all_context.append(ctx)

    return all_context


def fetch_graph_visual_data(driver, limit=100):
    """
    Fetch all nodes and relationships from Neo4j for visualization.
    Returns (nodes, edges) where:
        nodes: list of dicts with 'name' and 'type'
        edges: list of dicts with 'source', 'target', and 'relation'
    """
    nodes = []
    edges = []
    seen_nodes = set()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.name AS source, a.type AS source_type,
                   b.name AS target, b.type AS target_type,
                   r.relation AS relation
            LIMIT $limit
            """,
            limit=limit
        )
        for record in result:
            src = record["source"]
            tgt = record["target"]
            if src not in seen_nodes:
                seen_nodes.add(src)
                nodes.append({"name": src, "type": record["source_type"] or "Entity"})
            if tgt not in seen_nodes:
                seen_nodes.add(tgt)
                nodes.append({"name": tgt, "type": record["target_type"] or "Entity"})
            edges.append({
                "source": src,
                "target": tgt,
                "relation": record["relation"]
            })

    return nodes, edges
