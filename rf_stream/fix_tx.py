with open("rf_stream_tx_step5phy.py", "r") as f:
    content = f.read()

bad_chunk = """    # fragment to chunks so each packet has bounded payload size
    if cfg.ref_len > 0:
        rng = np.random.default_rng(cfg.ref_seed)
        data = rng.integers(0, 256, size=cfg.ref_len, dtype=np.uint8).tobytes()
        print(f"[TX] Generating reference payload: {len(data)} bytes (seed={cfg.ref_seed})")

    if chunk_bytes <= 0:"""
    
good_chunk = """    # fragment to chunks so each packet has bounded payload size
    if chunk_bytes <= 0:"""

content = content.replace(bad_chunk, good_chunk)

with open("rf_stream_tx_step5phy.py", "w") as f:
    f.write(content)
