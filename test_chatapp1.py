import asyncio

async def test_agent():
    import sys
    sys.path.append("/Users/uw-user/Desktop/wydot_cloud/.claude/worktrees/quizzical-antonelli")
    from chatapp1 import get_agent_executor, search_wydot_documents, _CURRENT_TOOL_SOURCES
    
    print("Testing Agent Executor...")
    agent = get_agent_executor(use_gemini=False)
    
    question = "Compare the 2010 rules for asphalt with the 2021 specs."
    print(f"\nQuestion: {question}\n")
    
    res = await agent.ainvoke({"input": question, "chat_history": []})
    
    print("\n--- Final Answer ---")
    print(res.get("output"))
    
    print("\n--- Sources Used ---")
    for s in _CURRENT_TOOL_SOURCES:
        print(f"[{s['year']}] {s['title']}")

if __name__ == "__main__":
    asyncio.run(test_agent())
