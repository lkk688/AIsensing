#based on sdradi/sim_video_e2e_asyncv2_lab.py
#!/usr/bin/env python3
# pluto_tx2rx2_cable_link_test.py
import argparse
import time
import numpy as np

from sdr_video_commv2_lab import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

def _bit_ber(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 1.0
    return float(np.sum(a[:n] != b[:n])) / float(n)


def pick_io(link):
    """
    返回 (tx_func, rx_func, name_tx, name_rx)
    尽量兼容你工程里 SDRVideoLink 的真实实现。
    """
    # 1) link 自己的
    if hasattr(link, "tx") and callable(getattr(link, "tx")):
        txf, txn = link.tx, "link.tx"
    elif hasattr(link, "send") and callable(getattr(link, "send")):
        txf, txn = link.send, "link.send"
    else:
        txf, txn = None, "None"

    if hasattr(link, "rx") and callable(getattr(link, "rx")):
        rxf, rxn = link.rx, "link.rx"
    elif hasattr(link, "receive") and callable(getattr(link, "receive")):
        rxf, rxn = link.receive, "link.receive"
    elif hasattr(link, "read_samples") and callable(getattr(link, "read_samples")):
        rxf, rxn = link.read_samples, "link.read_samples"
    else:
        rxf, rxn = None, "None"

    # 2) 常见：link.sdr / link.pluto / link.dev 里才是真正 adi.Pluto
    for dev_name in ["sdr", "pluto", "dev", "device"]:
        dev = getattr(link, dev_name, None)
        if dev is None:
            continue

        # TX
        if txf is None:
            if hasattr(dev, "tx") and callable(getattr(dev, "tx")):
                txf, txn = dev.tx, f"{dev_name}.tx"
            elif hasattr(dev, "tx_push") and callable(getattr(dev, "tx_push")):
                txf, txn = dev.tx_push, f"{dev_name}.tx_push"

        # RX
        if rxf is None:
            if hasattr(dev, "rx") and callable(getattr(dev, "rx")):
                rxf, rxn = dev.rx, f"{dev_name}.rx"
            elif hasattr(dev, "rx_read") and callable(getattr(dev, "rx_read")):
                rxf, rxn = dev.rx_read, f"{dev_name}.rx_read"

    return txf, rxf, txn, rxn


def iq_stats(x: np.ndarray):
    if x is None:
        return None
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0}
    rms = float(np.sqrt(np.mean(np.abs(x)**2)))
    peak = float(np.max(np.abs(x)))
    return {"n": int(x.size), "rms": rms, "peak": peak}

def main():
    ap = argparse.ArgumentParser(description="Minimal Pluto TX2<->RX2 (ch=1) cable OFDM link test")
    ap.add_argument("--uri", type=str, default="ip:192.168.3.2", help="Pluto URI (ip:xxx or usb:...)")
    ap.add_argument("--config", type=str, default="sdr_tuned_config.json", help="SDRConfig json used by your project")
    ap.add_argument("--fc", type=float, default=2.405e9, help="Center frequency Hz")
    ap.add_argument("--fs", type=float, default=3e6, help="Sample rate")
    ap.add_argument("--bw", type=float, default=3e6, help="RF bandwidth")
    ap.add_argument("--tx_gain", type=float, default=-20.0, help="TX gain dB")
    ap.add_argument("--rx_gain", type=float, default=20.0, help="RX gain dB")
    ap.add_argument("--tx_ch", type=int, default=1, help="TX channel index (TX2 -> 1)")
    ap.add_argument("--rx_ch", type=int, default=1, help="RX channel index (RX2 -> 1)")
    ap.add_argument("--tx_port", type=int, default=1, help="RF port select (your project mapping)")
    ap.add_argument("--rx_port", type=int, default=1, help="RF port select (your project mapping)")
    ap.add_argument("--npackets", type=int, default=200, help="How many packets to send")
    ap.add_argument("--payload_bits", type=int, default=2048, help="Bits per packet (payload)")
    ap.add_argument("--settle", type=float, default=0.25, help="Tuning settle seconds")
    ap.add_argument("--sleep_factor", type=float, default=1.0, help="TX pacing factor (>=1 safer)")
    ap.add_argument("--sync_threshold", type=float, default=35.0, help="OFDM sync threshold")
    ap.add_argument("--dlb", action="store_true", help="Enable digital loopback (for sanity check)")
    ap.add_argument("--no_fec", action="store_true", help="Disable FEC (recommended for raw BER test)")
    args = ap.parse_args()

    # ---- Build SDRConfig from your json, then override the essentials
    sdr_cfg = SDRConfig.load_from_json(args.config)
    # These attribute names must match your SDRConfig dataclass; adjust if yours differs:
    sdr_cfg.rx_uri = args.uri if hasattr(sdr_cfg, "rx_uri") else getattr(sdr_cfg, "rx_uri", args.uri)
    if hasattr(sdr_cfg, "sdr_ip"):
        sdr_cfg.sdr_ip = args.uri
    if hasattr(sdr_cfg, "fc"): sdr_cfg.fc = float(args.fc)
    if hasattr(sdr_cfg, "fs"): sdr_cfg.fs = float(args.fs)
    if hasattr(sdr_cfg, "bandwidth"): sdr_cfg.bandwidth = float(args.bw)
    if hasattr(sdr_cfg, "tx_gain"): sdr_cfg.tx_gain = float(args.tx_gain)
    if hasattr(sdr_cfg, "rx_gain"): sdr_cfg.rx_gain = float(args.rx_gain)

    # ---- OFDM config
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = float(args.sync_threshold)

    # ---- FEC: OFF by default for “最简单直连链路”
    if args.no_fec:
        fec_cfg = FECConfig(enabled=False, fec_type=FECType.NONE)
    else:
        # 你也可以改成 LDPC，但最简单链路建议先关掉，看纯 PHY
        fec_cfg = FECConfig(enabled=False, fec_type=FECType.NONE)

    link = SDRVideoLink(
        sdr_config=sdr_cfg,
        fec_config=fec_cfg,
        ofdm_config=ofdm_cfg,
        simulation_mode=False,   # <-- IMPORTANT: real Pluto
    )

    txf, rxf, txn, rxn = pick_io(link)
    print(f"[DBG] TX func = {txn}")
    print(f"[DBG] RX func = {rxn}")
    if rxf is None:
        raise RuntimeError("No RX function found. Your SDRVideoLink doesn't expose rx().")

    # 尝试把 RX buffer 调大（如果底层对象支持）
    for dev_name in ["sdr", "pluto", "dev", "device"]:
        dev = getattr(link, dev_name, None)
        if dev is None:
            continue
        if hasattr(dev, "rx_buffer_size"):
            try:
                dev.rx_buffer_size = 262144  # 先固定一个大点的
                print(f"[DBG] set {dev_name}.rx_buffer_size=262144")
            except Exception as e:
                print(f"[DBG] set rx_buffer_size failed on {dev_name}: {e}")

    # ---- Apply channel/port selection (尽量兼容你现有 link 的字段/方法)
    # 这些 setter 名称你项目里可能不同：如果报错，就按你的实现改这几行。
    if hasattr(link, "tx_ch"): link.tx_ch = int(args.tx_ch)
    if hasattr(link, "rx_ch"): link.rx_ch = int(args.rx_ch)
    if hasattr(link, "tx_port"): link.tx_port = int(args.tx_port)
    if hasattr(link, "rx_port"): link.rx_port = int(args.rx_port)

    # 如果你的 SDRVideoLink 内部封装了 iio 属性（比如 link.sdr 或 link.pluto）
    # 并且有 "loopback / port" 的设置接口，可以在这里设置：
    try:
        if hasattr(link, "set_loopback"):
            link.set_loopback(1 if args.dlb else 0)
        elif hasattr(link, "loopback"):
            link.loopback = 1 if args.dlb else 0
    except Exception:
        pass

    # 一些设备需要一点 settle
    time.sleep(float(args.settle))

    # ---- Determine how many bits fit per OFDM frame in your implementation
    if link.waveform != WaveformType.OFDM:
        print(f"[WARN] link.waveform={link.waveform}, expected OFDM. Continue anyway.")

    bits_per_frame = getattr(link.ofdm_config, "bits_per_frame", None)
    samples_per_frame = getattr(link.ofdm_config, "samples_per_frame", None)
    if bits_per_frame is None or samples_per_frame is None:
        raise RuntimeError("Your OFDMConfig doesn't expose bits_per_frame / samples_per_frame; adapt this script accordingly.")

    preamble = link._generate_preamble()
    preamble_len = len(preamble)

    print("==== Pluto Cable Link Test (TX2<->RX2) ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_gain={args.tx_gain} rx_gain={args.rx_gain}  tx_ch={args.tx_ch} rx_ch={args.rx_ch}  tx_port={args.tx_port} rx_port={args.rx_port}")
    print(f"[OFDM] bits_per_frame={bits_per_frame} samples_per_frame={samples_per_frame} preamble_len={preamble_len} sync_th={args.sync_threshold}")
    print(f"[MODE] {'DLB' if args.dlb else 'RF cable'}  FEC={'OFF' if not fec_cfg.enabled else fec_cfg.fec_type}")

    total_bits = 0
    total_bit_err = 0
    pkt_ok = 0
    pkt_fail = 0

    # RX buffer: read blocks and try sync each time
    # 经验：用一个稍大窗口，避免 sync “incomplete”
    rx_window_frames = 6
    rx_window = preamble_len + rx_window_frames * samples_per_frame

    for p in range(int(args.npackets)):
        # ---- generate payload bits
        tx_bits = np.random.randint(0, 2, size=int(args.payload_bits), dtype=np.uint8)

        # ---- pad to whole OFDM frames
        nframes = int(np.ceil(len(tx_bits) / bits_per_frame))
        pad = nframes * bits_per_frame - len(tx_bits)
        if pad > 0:
            tx_bits_padded = np.concatenate([tx_bits, np.zeros(pad, dtype=np.uint8)])
        else:
            tx_bits_padded = tx_bits

        # ---- FEC encode (if enabled) else raw
        if fec_cfg.enabled:
            tx_fec_bits = link.fec_codec.encode(tx_bits_padded)
        else:
            tx_fec_bits = tx_bits_padded

        # ---- Modulate -> samples
        tx_samples = link.transmit(tx_fec_bits)
        # normalize to avoid saturation; cable直连也建议保守点
        m = np.max(np.abs(tx_samples))
        if m > 0:
            tx_samples = tx_samples / m

        # ---- push to device (your SDRVideoLink.transmit likely already sends)
        # 你项目里 transmit() 可能只是生成 samples；真正送硬件是另一层。
        # 如果你的实现是 “transmit() 已经写入硬件”，那下面就不需要。
        if hasattr(link, "tx") and callable(getattr(link, "tx")):
            # e.g., link.tx(samples)
            link.tx(tx_samples)
        elif hasattr(link, "send") and callable(getattr(link, "send")):
            link.send(tx_samples)
        else:
            # fallback: assume link.transmit already transmitted through hardware TX path
            pass

        # pacing (让 RX 有机会读到完整包)
        time.sleep((len(tx_samples) / float(args.fs)) * float(args.sleep_factor))

        # ---- RX capture
        # 同样：你项目里可能是 link.rx(N) / link.receive(N) 之类
        rx = None
        if hasattr(link, "rx") and callable(getattr(link, "rx")):
            rx = link.rx(int(rx_window))
        elif hasattr(link, "receive") and callable(getattr(link, "receive")):
            rx = link.receive(int(rx_window))
        elif hasattr(link, "read_samples") and callable(getattr(link, "read_samples")):
            rx = link.read_samples(int(rx_window))

        if rx is None or len(rx) < preamble_len + samples_per_frame:
            pkt_fail += 1
            print(f"[{p:04d}] RX empty/short")
            continue

        st = iq_stats(rx)
        print(f"[DBG] RX got: {st}")
        if rx is None or st is None or st.get("n", 0) < 2000:
            pkt_fail += 1
            continue

        # ---- synchronize
        synced, met = link._synchronize(rx)
        if not met.get("sync_success", False):
            pkt_fail += 1
            print(f"[{p:04d}] sync fail  peak={met.get('peak_val',0):.1f}  incomplete={met.get('incomplete',False)}")
            continue

        # payload start (if provided)
        payload_start = met.get("payload_start", 0)
        # pull enough samples for this packet
        need_payload = nframes * samples_per_frame
        if payload_start + need_payload > len(synced):
            pkt_fail += 1
            print(f"[{p:04d}] sync ok but payload incomplete")
            continue
        payload = synced[payload_start:payload_start + need_payload]

        # ---- demodulate
        rx_fec_bits, _ = link.transceiver.demodulate(payload)

        # ---- FEC decode (if enabled)
        if fec_cfg.enabled:
            try:
                rx_bits = link.fec_codec.decode(rx_fec_bits)
            except Exception:
                pkt_fail += 1
                print(f"[{p:04d}] FEC decode fail")
                continue
        else:
            rx_bits = rx_fec_bits

        # ---- truncate to original payload bits
        rx_bits = rx_bits[:len(tx_bits_padded)]
        ber = _bit_ber(tx_bits_padded, rx_bits)

        # per-packet stats
        bit_err = int(np.sum(tx_bits_padded != rx_bits[:len(tx_bits_padded)]))
        total_bit_err += bit_err
        total_bits += int(len(tx_bits_padded))
        if bit_err == 0:
            pkt_ok += 1
        else:
            pkt_fail += 1

        if (p % 10) == 0 or bit_err != 0:
            print(f"[{p:04d}] ok={pkt_ok} fail={pkt_fail}  pktBER={ber:.3e}  peak={met.get('peak_val',0):.1f} cfo={met.get('cfo_est',0):.1f}Hz")

    # ---- Summary
    ber_total = (total_bit_err / total_bits) if total_bits > 0 else 1.0
    per = (pkt_fail / max(1, (pkt_ok + pkt_fail)))
    print("\n==== SUMMARY ====")
    print(f"Packets: ok={pkt_ok} fail={pkt_fail}  PER={per:.3f}")
    print(f"Bits: total={total_bits} err={total_bit_err}  BER={ber_total:.3e}")
    print("Tip: 若 RF cable PER/BER 差，但 DLB (--dlb) 很好 => 基本就是端口映射/线缆/增益/饱和/串扰问题。")

if __name__ == "__main__":
    main()