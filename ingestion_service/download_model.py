import os
from langchain_huggingface import HuggingFaceEmbeddings

print("⬇️ Downloading embedding model to bake into container...")
# This will download the model to the local huggingface cache
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Model downloaded successfully.")
