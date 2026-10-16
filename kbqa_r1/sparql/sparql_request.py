import requests
import json
from typing import List, Dict, Any, Optional

def execute_sparql(queries: List[str], url: str = "http://localhost:8000/execute") -> Dict[str, Any]:
    """
    Execute SPARQL queries using the SPARQL execution server.
    
    Args:
        queries: List of SPARQL queries to execute
        url: URL of the SPARQL execution server
        
    Returns:
        Dictionary containing the results of the SPARQL queries
    """
    try:
        response = requests.post(url, json={"queries": queries})
        return response.json()
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    queries = [
        """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?name WHERE {
            ns:m.02mjmr ns:type.object.name ?name .
        }
        """
    ]
    
    results = execute_sparql(queries)
    print(json.dumps(results, indent=2)) 