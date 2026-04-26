with open("rf_stream_rx_step5phy_v2.py", "r") as f:
    content = f.read()

import re

new_try_demod_at = """    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, int, bytes, dict, dict, dict]:
        \"\"\"
        Demod starting at stf_idx (index within rxw).
        \"\"\"
        ltf0 = stf_idx + stf_len
        
        # 1. Estimate CFO from STF (period=16 samples)
        cfo_hz = 0.0
        stf_start = stf_idx + 16
        if stf_start + 64 <= rxw.shape[0]:
            p0 = rxw[stf_start : stf_start + 48]
            p1 = rxw[stf_start + 16 : stf_start + 64]
            angle = float(np.angle(np.sum(p1 * np.conj(p0))))
            cfo_hz = angle / (2 * np.pi * (16.0 / cfg.fs))
            
        def extract_sym(start):
            if start + SYMBOL_LEN > rxw.shape[0]:
                return None
            td = rxw[start + N_CP : start + SYMBOL_LEN]
            t = np.arange(start + N_CP, start + SYMBOL_LEN) / cfg.fs
            td = td * np.exp(-1j * 2 * np.pi * cfo_hz * t)
            return np.fft.fftshift(np.fft.fft(td))

        # fine sweep around ltf0 to maximize channel "flatness quality"
        best_q = -1.0
        best_off = 0
        used_bins = np.array([sc_to_bin(k) for k in range(-26,27) if k!=0], dtype=int)

        for off in range(-cfg.ltf_off_sweep, cfg.ltf_off_sweep+1):
            Y = extract_sym(ltf0 + off)
            if Y is None:
                continue
            Ht = np.zeros(N_FFT, dtype=np.complex64)
            for k in range(-26,27):
                if k == 0:
                    continue
                idx = sc_to_bin(k)
                if np.abs(ltf_freq_ref[idx]) > 1e-6:
                    Ht[idx] = Y[idx] / ltf_freq_ref[idx]
            m = np.abs(Ht[used_bins])
            qv = float((np.mean(m)**2) / (np.var(m) + 1e-12))
            if qv > best_q:
                best_q = qv
                best_off = off

        ltf_start = ltf0 + best_off
        
        # LTF H estimate
        Ys = []
        for i in range(cfg.ltf_symbols):
            Y = extract_sym(ltf_start + i*SYMBOL_LEN)
            if Y is None: break
            Ys.append(Y)
        H = None
        if Ys:
            Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
            H = np.ones(N_FFT, dtype=np.complex64)
            for k in range(-26,27):
                if k == 0: continue
                idx = sc_to_bin(k)
                if np.abs(ltf_freq_ref[idx]) > 1e-6:
                    H[idx] = Yavg[idx] / ltf_freq_ref[idx]

        ltf_info = {"score": best_q, "H": H, "Y0": None, "Y1": None} # simplify Y0/Y1
        if H is None:
            return False, "ltf_fail", -1, -1, b"", {"ltf_q": best_q, "ltf_start": ltf_start}, None, ltf_info

        payload_start = ltf_start + cfg.ltf_symbols*SYMBOL_LEN

        # probe demod
        pilot_vals = np.array([1,1,1,-1], dtype=np.complex64)
        phase_acc = 0.0
        freq_acc = 0.0
        prev_phase = None

        data_syms_all = []
        evm_log = []
        pilot_pwr_log = []
        phase_err_log = []
        phase_acc_log = []
        freq_acc_log = []
        all_data_pre = []
        all_data_post = []

        for si in range(cfg.max_ofdm_syms_cap):
            Y = extract_sym(payload_start + si*SYMBOL_LEN)
            if Y is None:
                break"""

start_idx = content.find("    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, int, bytes, dict, dict, dict]:")
end_idx = content.find("            Ye = Y.copy()", start_idx)

content = content[:start_idx] + new_try_demod_at + "\n" + content[end_idx:]

with open("rf_stream_rx_step5phy_v2.py", "w") as f:
    f.write(content)
