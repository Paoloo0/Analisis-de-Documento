import sys
import time

print("Testing fastembed loading...")
try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("[+] FastEmbedEmbeddings imported successfully!")
    
    texts = ["Hola mundo", "Esto es una prueba de embeddings rápidos y ligeros sin PyTorch"]
    start = time.time()
    vectors = embeddings.embed_documents(texts)
    duration = time.time() - start
    
    print(f"[+] Embeddings generated successfully in {duration:.4f} seconds!")
    print(f"[+] Vector length: {len(vectors[0])}")
except Exception as e:
    print(f"[-] Error: {e}")
