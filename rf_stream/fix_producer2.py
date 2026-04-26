with open("rf_stream_tx_step5phy.py", "r") as f:
    content = f.read()

bad_chunk = """    seq = 0
    total = len(chunks)
    while not stop_ev.is_set():
        for ch in chunks:"""

good_chunk = """    total = len(chunks)
    while not stop_ev.is_set():
        seq = 0
        for ch in chunks:"""
content = content.replace(bad_chunk, good_chunk)

with open("rf_stream_tx_step5phy.py", "w") as f:
    f.write(content)
