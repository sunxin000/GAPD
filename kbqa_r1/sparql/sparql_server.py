# import json
# import os
# import warnings
# from typing import List, Dict, Optional
# import argparse
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# import torch
# import numpy as np
# from tqdm import tqdm

# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from SPARQLWrapper import SPARQLWrapper, JSON

# parser = argparse.ArgumentParser(description="Launch the SPARQL execution server.")
# parser.add_argument("--sparql_endpoint", type=str, default="http://localhost:3001/sparql", help="SPARQL endpoint URL.")
# parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
# parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of concurrent workers for SPARQL execution.")

# args = parser.parse_args()

# app = FastAPI()

# # Global thread pool for SPARQL execution
# executor = ThreadPoolExecutor(max_workers=args.max_workers)

# def execute_sparql(sparql_query):
#     """Execute a SPARQL query and return the results."""
#     try:
#         sparql = SPARQLWrapper(args.sparql_endpoint)
#         print(f"Executing SPARQL query: {sparql_query}")
#         sparql.setQuery(sparql_query)
#         sparql.setReturnFormat(JSON)
#         results = sparql.query().convert()
#         print(f"Results: {results}")
#         return results["results"]["bindings"]
#     except Exception as e:
#         print(f"Error executing SPARQL query: {e}")
#         return {"error": str(e)}

# def process_results(results):
#     """Process SPARQL results to a more readable format."""
#     processed_results = []
    
#     if isinstance(results, dict) and "error" in results:
#         return [{"error": results["error"]}]
    
#     for result in results:
#         processed_result = {}
#         for var, value in result.items():
#             processed_value = value["value"]
#             # Remove Freebase namespace prefix if present
#             if processed_value.startswith("http://rdf.freebase.com/ns/"):
#                 processed_value = processed_value.replace("http://rdf.freebase.com/ns/", "")
#             processed_result[var] = processed_value
#         processed_results.append(processed_result)
    
#     return processed_results

# class SPARQLRequest(BaseModel):
#     queries: List[str]

# async def execute_single_query(query: str) -> Dict:
#     """Execute a single SPARQL query asynchronously."""
#     loop = asyncio.get_event_loop()
#     try:
#         # Execute SPARQL query in thread pool
#         raw_results = await loop.run_in_executor(executor, execute_sparql, query)
        
#         # Process the results
#         processed_results = process_results(raw_results)
        
#         return {
#             "query": query,
#             "results": processed_results
#         }
#     except Exception as e:
#         return {
#             "query": query,
#             "error": str(e)
#         }

# @app.post("/execute")
# async def execute_endpoint(request: SPARQLRequest):
#     """Endpoint to execute SPARQL queries with concurrent processing."""
#     if not request.queries:
#         raise HTTPException(status_code=400, detail="No queries provided")
    
#     print(f"Processing {len(request.queries)} SPARQL queries with up to {args.max_workers} concurrent workers")
    
#     # Execute queries concurrently
#     tasks = [execute_single_query(query) for query in request.queries]
#     results = await asyncio.gather(*tasks)
    
#     print(f"Completed processing {len(results)} SPARQL queries")
    
#     return {"results": results}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {"status": "healthy", "max_workers": args.max_workers}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=args.port) 