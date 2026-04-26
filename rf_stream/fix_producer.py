with open("rf_stream_tx_step5phy.py", "r") as f:
    content = f.read()

bad_chunk = """    seq = 0
    total = len(chunks)
    for ch in chunks:
        if stop_ev.is_set():
            break"""

good_chunk = """    seq = 0
    total = len(chunks)
    while not stop_ev.is_set():
        for ch in chunks:
            if stop_ev.is_set():
                break"""
content = content.replace(bad_chunk, good_chunk)

with open("rf_stream_tx_step5phy.py", "w") as f:
    f.write(content)
