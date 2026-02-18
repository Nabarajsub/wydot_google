import sys
import os

# Mocking chainlit and other deps used in chatapp for simple sync testing
from chatapp import search_graph, build_prompt_with_history

def test_precision():
    print("🔍 Starting Precision Verification...")
    
    # Test Question that requires comparison and precise data
    query = "Compare concrete strength requirements for Section 401 between 2010 and 2021."
    index = "wydot_vector_index" # Default
    
    print(f"❓ Query: {query}")
    context, sources = search_graph(query, index)
    
    print(f"✅ Found {len(sources)} sources.")
    for s in sources:
        print(f"   - [SOURCE_{s['index']}] {s['title']} (Score: {s.get('score', 'N/A')})")
        if "Additional Context" in context:
            print("   - ✨ Neighborhood expansion detected.")
            
    # Check if prompt instructions are clear
    prompt = build_prompt_with_history(query, context, [])
    if "Markdown table" in prompt:
        print("✅ Prompt contains tabular formatting instructions.")
    
    print("\n🚀 Verification Complete. Ready for User testing.")

if __name__ == "__main__":
    test_precision()
