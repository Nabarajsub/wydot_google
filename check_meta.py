import asyncio
from utils.database_manager import get_retriever, NEO4J_INDEX_DEFAULT
from dotenv import load_dotenv

load_dotenv()

ret = get_retriever(NEO4J_INDEX_DEFAULT)
docs = ret.invoke("asphalt")
for d in docs[:1]:
    print("Metadata keys:", d.metadata.keys())
    print("Metadata values:", d.metadata)
