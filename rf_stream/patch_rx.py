import re

with open("rf_stream_rx_step5phy_v2.py", "r") as f:
    content = f.read()

# 1. Update parse_packet_bytes
parse_old = """def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    \"\"\"
    returns: (ok, reason, seq, payload)
    \"\"\"
    if len(bb) < 4+4+2+4:
        return False, "too_short", -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, b""
    seq = int.from_bytes(bb[4:8], "little")
    plen = int.from_bytes(bb[8:10], "little")
    need = 10 + plen + 4
    if len(bb) < need:
        return False, "need_more", seq, b""
    body = bb[:10+plen]
    crc_rx = int.from_bytes(bb[10+plen:10+plen+4], "little")
    crc_ok = (zlib.crc32(body) & 0xFFFFFFFF) == crc_rx
    if not crc_ok:
        return False, "crc_fail", seq, b""
    payload = bb[10:10+plen]
    return True, "ok", seq, payload"""

parse_new = """def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, int, bytes]:
    \"\"\"
    returns: (ok, reason, seq, total, payload)
    \"\"\"
    if len(bb) < 14:
        return False, "too_short", -1, -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, -1, b""
    seq = int.from_bytes(bb[4:6], "little")
    total = int.from_bytes(bb[6:8], "little")
    plen = int.from_bytes(bb[8:10], "little")
    need = 10 + plen + 4
    if len(bb) < need:
        return False, "need_more", seq, total, b""
    body = bb[:10+plen]
    crc_rx = int.from_bytes(bb[10+plen:10+plen+4], "little")
    crc_ok = (zlib.crc32(body) & 0xFFFFFFFF) == crc_rx
    if not crc_ok:
        return False, "crc_fail", seq, total, b""
    payload = bb[10:10+plen]
    return True, "ok", seq, total, payload"""

content = content.replace(parse_old, parse_new)

# 2. Add figure worker, make_reference_payload, and ber_bits
helpers_add = """

def make_reference_payload(seed: int, length: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()

def ber_bits(a: bytes, b: bytes) -> tuple[int,int,float]:
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0, 0.0
    aa = np.unpackbits(np.frombuffer(a[:n], dtype=np.uint8))
    bb = np.unpackbits(np.frombuffer(b[:n], dtype=np.uint8))
    err = int(np.sum(aa != bb))
    tot = int(len(aa))
    return err, tot, float(err/(tot+1e-12))

def figure_worker_thread(stop_ev: threading.Event, fig_q: queue.Queue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[FIG] Figure worker started.")
    while not stop_ev.is_set():
        try:
            item = fig_q.get(timeout=0.5)
        except queue.Empty:
            continue
            
        (path, rx, fs, tone_info, corr, idxs, peaks, chosen, ltf_info, demod_info, parse_info, title) = item
        try:
            stf_idx = chosen.get("stf_idx", -1)
            stf_ncc = chosen.get("stf_ncc", 0.0)

            fig = plt.figure(figsize=(18, 10))

            # 1) time |rx|
            ax = fig.add_subplot(3,4,1)
            N = min(len(rx), 80000)
            ax.plot(np.abs(rx[:N]))
            if stf_idx >= 0 and stf_idx < N:
                ax.axvline(stf_idx, color="r", linestyle="--", label="chosen STF")
            ax.set_title("Time |rx| (first 80k)")
            ax.grid(True)
            ax.legend(loc="upper right")

            # 2) tone FFT
            ax = fig.add_subplot(3,4,2)
            if tone_info is not None:
                freqs, db, detected, pk_db = tone_info
                ax.plot(freqs/1e3, db)
                ax.set_title(f"Tone FFT (dB)\\ndetected={detected/1e3:.1f}kHz peak={pk_db:.1f}dB")
                ax.grid(True)
            else:
                ax.axis("off")

            # 3) STF NCC corr
            ax = fig.add_subplot(3,4,3)
            if len(corr) > 0:
                ax.plot(idxs, corr)
                for s,v in peaks:
                    ax.axvline(s, color="k", alpha=0.12)
                if stf_idx >= 0:
                    ax.axvline(stf_idx, color="r", linestyle="--", label=f"chosen {stf_ncc:.3f}")
                ax.set_title("STF NCC corr")
                ax.grid(True)
                ax.legend()
            else:
                ax.axis("off")

            # 4) LTF mags Y0/Y1
            ax = fig.add_subplot(3,4,4)
            if ltf_info is not None and ltf_info.get("Y0") is not None:
                ax.plot(np.abs(ltf_info["Y0"]), label="|Y0|")
                ax.plot(np.abs(ltf_info["Y1"]), label="|Y1|", alpha=0.8)
                ax.set_title(f"LTF FFT mags (2 repeats)\\nscore={ltf_info.get('score',0):.3f}")
                ax.grid(True)
                ax.legend()
            else:
                ax.axis("off")

            # 5) |H|
            ax = fig.add_subplot(3,4,5)
            if ltf_info is not None and ltf_info.get("H") is not None:
                ax.plot(np.abs(ltf_info["H"]))
                ax.set_title("|H| (fftshift bins)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 6) angle(H)
            ax = fig.add_subplot(3,4,6)
            if ltf_info is not None and ltf_info.get("H") is not None:
                ax.plot(np.unwrap(np.angle(ltf_info["H"])))
                ax.set_title("angle(H) unwrap")
                ax.grid(True)
            else:
                ax.axis("off")

            # 7) pilot power
            ax = fig.add_subplot(3,4,7)
            if demod_info is not None and demod_info.get("pilot_pwr") is not None and len(demod_info["pilot_pwr"]) > 0:
                ax.plot(demod_info["pilot_pwr"])
                ax.set_title("Pilot power per OFDM symbol")
                ax.grid(True)
            else:
                ax.axis("off")

            # 8) pilot loop traces
            ax = fig.add_subplot(3,4,8)
            if demod_info is not None and demod_info.get("phase_err") is not None and len(demod_info["phase_err"]) > 0:
                ax.plot(demod_info["phase_err"], label="phase_err (unwrap)")
                ax.plot(demod_info["phase_acc"], label="phase_acc")
                ax.plot(demod_info["freq_acc"], label="freq_acc")
                ax.set_title("Pilot loop traces")
                ax.grid(True)
                ax.legend()
            else:
                ax.axis("off")

            # 9) constellation pre/post CPE
            ax = fig.add_subplot(3,4,9)
            if demod_info is not None and demod_info.get("data_pre") is not None and len(demod_info["data_pre"]) > 0:
                pre = demod_info["data_pre"]
                post = demod_info["data_post"]
                ax.scatter(np.real(pre), np.imag(pre), s=4, alpha=0.35, label="pre-CPE")
                ax.scatter(np.real(post), np.imag(post), s=4, alpha=0.35, label="post-CPE")
                ax.set_title("Constellation (data bins)")
                ax.grid(True)
                ax.axis("equal")
                ax.legend()
            else:
                ax.axis("off")

            # 10) EVM per symbol
            ax = fig.add_subplot(3,4,10)
            if demod_info is not None and demod_info.get("evm_db") is not None and len(demod_info["evm_db"]) > 0:
                ax.plot(demod_info["evm_db"])
                ax.set_title("EVM per OFDM symbol (dB)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 11) angle histogram
            ax = fig.add_subplot(3,4,11)
            if demod_info is not None and demod_info.get("data_post") is not None and len(demod_info["data_post"]) > 0:
                ang = np.angle(demod_info["data_post"])
                ax.hist(ang, bins=60)
                ax.set_title("Angle hist (post-CPE)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 12) text box
            ax = fig.add_subplot(3,4,12)
            ax.axis("off")
            txt = []
            txt.append(title)
            txt.append(f"CFO_use={chosen.get('cfo_use',0):+.1f} Hz")
            txt.append(f"STF_idx={stf_idx}  STF_ncc={stf_ncc:.3f}  LTF_score={ltf_info.get('score',0) if ltf_info else 0:.3f}")
            if parse_info is None:
                txt.append("PARSE: FAIL")
            else:
                txt.append(f"PARSE: start={parse_info.get('start', -1)} ok={parse_info.get('ok', False)}")
                txt.append(f" seq={parse_info.get('seq', -1)} total={parse_info.get('total', -1)} plen={parse_info.get('plen', -1)}")
            if demod_info is not None and demod_info.get("nsyms", 0) > 0:
                txt.append(f"Demod: ofdm_syms={demod_info['nsyms']} bytes={demod_info.get('bytes_len',0)}")
            ax.text(0.02, 0.98, "\\n".join(txt), va="top", family="monospace")

            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
        except Exception as e:
            import traceback
            print(f"[FIG] Error saving figure {path}: {e}")
            traceback.print_exc()

    print("[FIG] Figure worker stopped.")
"""
content = content.replace("# =========================", helpers_add + "\n# =========================", 1)

# Update RxConfig
rx_cfg_old = """    save_npz: bool
    mode: str
    verbose: bool"""
rx_cfg_new = """    save_npz: bool
    mode: str
    verbose: bool
    ref_seed: int
    ref_len: int"""
content = content.replace(rx_cfg_old, rx_cfg_new)

# 3. Update dsp_thread parameters and logic
dsp_sig_old = """def dsp_thread(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: RxConfig):"""
dsp_sig_new = """def dsp_thread(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", fig_q: queue.Queue, cfg: RxConfig):"""
content = content.replace(dsp_sig_old, dsp_sig_new)

try_demod_old = """    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, bytes, dict]:"""
try_demod_new = """    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, int, bytes, dict, dict, dict]:"""
content = content.replace(try_demod_old, try_demod_new)

# Update return values of try_demod_at
demod_ret_old = """        H = channel_estimate_from_ltf(rxw, ltf_start, ltf_freq_ref, cfg.ltf_symbols)
        if H is None:
            return False, "ltf_fail", -1, b"", {"ltf_q": best_q, "ltf_start": ltf_start}"""
demod_ret_new = """        H = channel_estimate_from_ltf(rxw, ltf_start, ltf_freq_ref, cfg.ltf_symbols)
        ltf_info = {"score": best_q, "H": H, "Y0": None, "Y1": None} # simplify Y0/Y1
        if H is None:
            return False, "ltf_fail", -1, -1, b"", {"ltf_q": best_q, "ltf_start": ltf_start}, None, ltf_info"""
content = content.replace(demod_ret_old, demod_ret_new)

demod_ret2_old = """        if not data_syms_all:
            return False, "no_payload", -1, b"", {"ltf_q": best_q, "ltf_start": ltf_start, "payload_start": payload_start}"""
demod_ret2_new = """        demod_info = {
            "nsyms": len(data_syms_all),
            "pilot_pwr": np.array(pilot_pwr_log, dtype=np.float32),
            "phase_err": np.array(phase_err_log, dtype=np.float32),
            "phase_acc": np.array(phase_acc_log, dtype=np.float32),
            "freq_acc": np.array(freq_acc_log, dtype=np.float32),
            "evm_db": np.array(evm_log, dtype=np.float32),
            "data_pre": np.concatenate(all_data_pre) if len(all_data_pre) else np.array([]),
            "data_post": np.concatenate(all_data_post) if len(all_data_post) else np.array([]),
        }
        if not data_syms_all:
            return False, "no_payload", -1, -1, b"", {"ltf_q": best_q, "ltf_start": ltf_start, "payload_start": payload_start}, demod_info, ltf_info"""
content = content.replace(demod_ret2_old, demod_ret2_new)

# And the success return:
success_ret_old = """        ok, reason, seq, payload = parse_packet_bytes(bb)
        diag = {
            "ltf_q": best_q,
            "ltf_start": ltf_start,
            "payload_start": payload_start,
            "probe_evm": float(np.mean(evm_list)) if evm_list else 0.0,
        }
        return ok, reason, seq, payload, diag"""
success_ret_new = """        ok, reason, seq, total, payload = parse_packet_bytes(bb)
        diag = {
            "ltf_q": best_q,
            "ltf_start": ltf_start,
            "payload_start": payload_start,
            "probe_evm": float(np.mean(evm_list)) if evm_list else 0.0,
        }
        demod_info["bytes_len"] = len(bb)
        parse_info = {"ok": ok, "seq": seq, "total": total, "plen": len(payload)} if bb else None
        return ok, reason, seq, total, payload, diag, demod_info, ltf_info"""
content = content.replace(success_ret_old, success_ret_new)

# 4. Modify candidate testing loop in dsp_thread to use the new returns and submit figure:
test_loop_old = """            best_ok = False
            best_reason = "no_try"
            best_seq = -1
            best_payload = b""
            best_diag = {}
            best_stf = -1
            best_xc_peak = float(corr_norm[top_idx[0]])
            best_xc_idx = int(top_idx[0])

            for cand in top_idx:
                if corr_norm[cand] < cfg.xcorr_min_peak:
                    continue
                ok, reason, seq, payload, diag = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True
                    best_reason = "ok"
                    best_seq = int(seq)
                    best_payload = payload
                    best_diag = diag
                    best_stf = int(cand)
                    break
                else:
                    if best_stf < 0:
                        best_reason = reason
                        best_diag = diag
                        best_stf = int(cand)"""
test_loop_new = """            best_ok = False
            best_reason = "no_try"
            best_seq = -1
            best_total = -1
            best_payload = b""
            best_diag = {}
            best_stf = -1
            best_xc_peak = float(corr_norm[top_idx[0]])
            best_xc_idx = int(top_idx[0])
            best_demod = None
            best_ltf = None

            for cand in top_idx:
                if corr_norm[cand] < cfg.xcorr_min_peak:
                    continue
                ok, reason, seq, total, payload, diag, d_info, l_info = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True
                    best_reason = "ok"
                    best_seq = int(seq)
                    best_total = int(total)
                    best_payload = payload
                    best_diag = diag
                    best_stf = int(cand)
                    best_demod = d_info
                    best_ltf = l_info
                    break
                else:
                    if best_stf < 0:
                        best_reason = reason
                        best_diag = diag
                        best_stf = int(cand)
                        best_demod = d_info
                        best_ltf = l_info"""
content = content.replace(test_loop_old, test_loop_new)

# 5. Adding Figure Queue dispatch and Reassembly Logic
add_fig_old = """            if best_ok:
                good += 1
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")"""
add_fig_new = """            
            # Enqueue figure saving
            if best_stf >= 0:
                chosen = {"stf_idx": best_stf, "stf_ncc": best_xc_peak, "cfo_use": 0.0}
                parse_info = {"ok": best_ok, "seq": best_seq, "total": best_total, "plen": len(best_payload)}
                title = f"cap={cap} {'OK' if best_ok else 'FAIL'} {best_reason}"
                fig_path = os.path.join(cfg.save_dir, f"cap_{cap:06d}_{'ok' if best_ok else 'fail'}.png")
                fig_q.put((fig_path, rxw.copy(), cfg.fs, None, corr_norm, np.arange(len(corr_norm)), [(int(x), float(corr_norm[x])) for x in top_idx], chosen, best_ltf, best_demod, parse_info, title))

            if best_ok:
                good += 1
                got_packets[best_seq] = best_payload
                if total_expected is None and best_total > 0:
                    total_expected = best_total
                    
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")"""
content = content.replace(add_fig_old, add_fig_new)

# Need to add got_packets dictionary
init_good_old = """    cap = 0
    good = 0"""
init_good_new = """    cap = 0
    good = 0
    got_packets = {}
    total_expected = None"""
content = content.replace(init_good_old, init_good_new)

# Add BER logic after the loop
finally_old = """    finally:
        fcsv.flush()
        fcsv.close()
        print("[RX] DSP thread stopped. cap=", cap, "good=", good)"""
finally_new = """    finally:
        fcsv.flush()
        fcsv.close()
        print("[RX] DSP thread stopped. cap=", cap, "good=", good)
        if total_expected is not None and len(got_packets) >= total_expected:
            full = b"".join(got_packets[i] for i in range(total_expected) if i in got_packets)
            outf = os.path.join(cfg.save_dir, "recovered_payload.bin")
            with open(outf, "wb") as f:
                f.write(full)
            print(f"\\n[RX] Reassembled payload: {len(full)} bytes -> {outf}")
            if cfg.ref_len > 0:
                ref_payload = make_reference_payload(cfg.ref_seed, cfg.ref_len)
                err, tot, ber = ber_bits(full, ref_payload)
                print(f"[BER] compare_len={min(len(full),len(ref_payload))} bytes bit_err={err} bit_tot={tot} BER={ber:.3e}")
        else:
            print("\\n[RX] Incomplete payload reassembly.")
            if total_expected:
                print(f"  got {len(got_packets)}/{total_expected} packets")"""
content = content.replace(finally_old, finally_new)

# 6. Update arguments and threading
args_old = """    ap.add_argument("--mode", type=str, default="packet", choices=["packet","sweep"])
    ap.add_argument("--verbose", action="store_true")"""
args_new = """    ap.add_argument("--mode", type=str, default="packet", choices=["packet","sweep"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len", type=int, default=0)"""
content = content.replace(args_old, args_new)

cfg_old = """        save_npz=bool(args.save_npz),
        mode=args.mode,
        verbose=bool(args.verbose),
    )"""
cfg_new = """        save_npz=bool(args.save_npz),
        mode=args.mode,
        verbose=bool(args.verbose),
        ref_seed=args.ref_seed,
        ref_len=args.ref_len,
    )"""
content = content.replace(cfg_old, cfg_new)

threads_old = """    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg), daemon=True)

    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, cfg), daemon=True)

    t_acq.start()
    t_dsp.start()"""
threads_new = """    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
    fig_q = queue.Queue()
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg), daemon=True)
    t_fig = threading.Thread(target=figure_worker_thread, args=(stop_ev, fig_q), daemon=True)

    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, fig_q, cfg), daemon=True)

    t_acq.start()
    t_fig.start()
    t_dsp.start()"""
content = content.replace(threads_old, threads_new)

with open("rf_stream_rx_step5phy_v2.py", "w") as f:
    f.write(content)
