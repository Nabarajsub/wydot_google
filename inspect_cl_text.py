import chainlit as cl
import asyncio

async def test():
    t = cl.Text(name="Source 1", content="Hello world", display="side")
    print(t.to_dict())

if __name__ == "__main__":
    asyncio.run(test())
