import re

with open("rf_stream_tx_step5phy.py", "r") as f:
    content = f.read()

args_old = """    ap.add_argument("--chunk_bytes", type=int, default=512, help="fragment file/payload into chunks")"""
args_new = """    ap.add_argument("--chunk_bytes", type=int, default=512, help="fragment file/payload into chunks")
    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len", type=int, default=0)"""
content = content.replace(args_old, args_new)

thread_old = """            args=(stop_ev, q, cfg, args.infile, args.payload, args.payload_len, args.chunk_bytes),"""
thread_new = """            args=(stop_ev, q, cfg, args.infile, args.payload, args.payload_len, args.chunk_bytes, args.ref_seed, args.ref_len),"""
content = content.replace(thread_old, thread_new)

sig_old = """                     infile: str, payload_str: str, payload_len: int, chunk_bytes: int):"""
sig_new = """                     infile: str, payload_str: str, payload_len: int, chunk_bytes: int, ref_seed: int = 0, ref_len: int = 0):"""
content = content.replace(sig_old, sig_new)

gen_old = """    if infile:
        with open(infile, "rb") as f:
            data = f.read()
    elif payload_str:
        data = payload_str.encode("utf-8")
    else:
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()"""
gen_new = """    if ref_len > 0:
        rng = np.random.default_rng(ref_seed)
        data = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
        print(f"[TX] Generating reference payload: {len(data)} bytes (seed={ref_seed})")
    elif infile:
        with open(infile, "rb") as f:
            data = f.read()
    elif payload_str:
        data = payload_str.encode("utf-8")
    else:
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()"""
content = content.replace(gen_old, gen_new)

with open("rf_stream_tx_step5phy.py", "w") as f:
    f.write(content)
