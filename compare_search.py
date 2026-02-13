
import asyncio
import os
import sys
from chatapp import search_graph, search_graph_async, get_retriever

# Mock chainlit context if needed, or ensuring env vars are set
# The chatapp.py uses os.getenv for keys. Assuming they are set in the environment or .env

async def compare_search(query):
    print(f"Query: {query}")
    
    print("\n--- Sync Search ---")
    try:
        # Note: search_graph in chatapp.py imports asyncio and tries to use event loop
        # We need to bypass that check or ensure we call the logic we want.
        # Actually search_graph definition in chatapp.py:
        # ret = get_retriever(...)
        # docs = ret.invoke(query)
        # So it should work if we just call it.
        context_sync, sources_sync = search_graph(query, "wydot_vector_index")
        print(f"Sync Context Length: {len(context_sync)}")
        print(f"Sync Sources: {len(sources_sync)}")
        if sources_sync:
            print(f"First Source Sync: {sources_sync[0]['title']}")
    except Exception as e:
        print(f"Sync Failed: {e}")

    print("\n--- Async Search ---")
    try:
        context_async, sources_async = await search_graph_async(query, "wydot_vector_index")
        print(f"Async Context Length: {len(context_async)}")
        print(f"Async Sources: {len(sources_async)}")
        if sources_async:
            print(f"First Source Async: {sources_async[0]['title']}")
            
        if len(sources_sync) != len(sources_async):
            print("\n❌ MISMATCH in source count!")
        else:
            print("\n✅ Source counts match.")
            
    except Exception as e:
        print(f"Async Failed: {e}")

if __name__ == "__main__":
    query = "who is the governor of wyoming"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    asyncio.run(compare_search(query))
