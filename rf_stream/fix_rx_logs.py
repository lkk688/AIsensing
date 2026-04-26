with open("rf_stream_rx_step5phy_v2.py", "r") as f:
    content = f.read()

bad_chunk = """            phase_acc += freq_acc + cfg.kp * ph
            Ye *= np.exp(-1j*phase_acc).astype(np.complex64)

            ds = Ye[DATA_BINS]"""

good_chunk = """            phase_acc += freq_acc + cfg.kp * ph
            
            phase_err_log.append(ph)
            phase_acc_log.append(phase_acc)
            freq_acc_log.append(freq_acc)
            pilot_pwr_log.append(float(np.mean(np.abs(rp)**2)))
            all_data_pre.append(Ye[DATA_BINS].copy())
            
            Ye *= np.exp(-1j*phase_acc).astype(np.complex64)

            ds = Ye[DATA_BINS]
            all_data_post.append(ds.copy())"""

content = content.replace(bad_chunk, good_chunk)

with open("rf_stream_rx_step5phy_v2.py", "w") as f:
    f.write(content)
