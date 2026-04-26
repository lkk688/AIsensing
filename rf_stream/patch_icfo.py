import numpy as np

with open("rf_stream_rx_step5phy_v2.py", "r") as f:
    content = f.read()

old_ltf = """        # LTF H estimate
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
                    H[idx] = Yavg[idx] / ltf_freq_ref[idx]"""

new_ltf = """        # LTF ICFO and H estimate
        Ys = []
        for i in range(cfg.ltf_symbols):
            Y = extract_sym(ltf_start + i*SYMBOL_LEN)
            if Y is None: break
            Ys.append(Y)
        H = None
        if Ys:
            Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
            
            # Estimate ICFO
            best_icfo_score = -1.0
            best_dk = 0
            for dk in range(-15, 16):
                # shift ltf_freq_ref by dk
                ref_shifted = np.roll(ltf_freq_ref, dk)
                # only correlate on used bins (shifted)
                # actually it's easier to correlate Yavg with ref_shifted
                corr = np.abs(np.vdot(ref_shifted, Yavg))
                if corr > best_icfo_score:
                    best_icfo_score = corr
                    best_dk = dk
                    
            # Apply ICFO correction
            # Re-extract with total CFO!
            icfo_hz = best_dk * (cfg.fs / N_FFT)
            cfo_hz += icfo_hz
            
            Ys_new = []
            for i in range(cfg.ltf_symbols):
                Y = extract_sym(ltf_start + i*SYMBOL_LEN)
                if Y is None: break
                Ys_new.append(Y)
                
            if Ys_new:
                Yavg = np.mean(np.stack(Ys_new, axis=0), axis=0)
                H = np.ones(N_FFT, dtype=np.complex64)
                for k in range(-26,27):
                    if k == 0: continue
                    idx = sc_to_bin(k)
                    if np.abs(ltf_freq_ref[idx]) > 1e-6:
                        H[idx] = Yavg[idx] / ltf_freq_ref[idx]"""

content = content.replace(old_ltf, new_ltf)

with open("rf_stream_rx_step5phy_v2.py", "w") as f:
    f.write(content)
