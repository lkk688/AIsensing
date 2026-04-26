with open("rf_stream_tx_step5phy.py", "r") as f:
    content = f.read()

import re
# Find the start of the while loop
start_idx = content.find("    total = len(chunks)\n    while not stop_ev.is_set():\n        seq = 0\n        for ch in chunks:")
end_idx = content.find("    print(\"[TX] producer done. (TX will continue idling)\")", start_idx)

good_chunk = """    total = len(chunks)
    while not stop_ev.is_set():
        seq = 0
        for ch in chunks:
            if stop_ev.is_set():
                break
            frame_bytes = build_packet_bytes(seq, total, ch)
            sig = bytes_to_ofdm_samples(
                frame_bytes, cfg.repeat, stf, ltf,
                fs=cfg.fs,
                tone_duration_ms=cfg.tone_duration_ms,
                tone_freq_hz=cfg.tone_freq_hz,
                gap_short=cfg.gap_short,
                gap_long=cfg.gap_long,
                tx_scale=cfg.tx_scale
            )
            sig = fit_to_fixed_len(sig, cfg.fixed_len)
            q.put(sig, block=True)
            print(f"[TX] enqueued packet seq={seq} payload={len(ch)}B frame_bytes={len(frame_bytes)} sig_len={len(sig)}")
            seq += 1

"""
content = content[:start_idx] + good_chunk + content[end_idx:]

with open("rf_stream_tx_step5phy.py", "w") as f:
    f.write(content)
