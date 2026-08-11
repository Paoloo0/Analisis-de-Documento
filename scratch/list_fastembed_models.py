from fastembed import TextEmbedding

print("Supported models in fastembed:")
for m in TextEmbedding.list_supported_models():
    print(f"- {m['model']} (dim: {m['dim']}, size_in_GB: {m.get('size_in_GB', 'N/A')}, description: {m.get('description', '')})")
