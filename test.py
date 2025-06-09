for i, chunk in enumerate(chunks[:3]):  # print first 3
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
