
import numpy as np

# IEEE 802.11a LTF sequence (52 subcarriers + DC=0)
# Indices -26 to 26
LTF_SEQ = np.array([
    1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, # -26 to -1
    0, # 0
    1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1 # 1 to 26
])

def get_ltf_val(k):
    if k < -26 or k > 26: return 0
    return LTF_SEQ[k + 26]

# Test autocorrelation
def test_autocorr():
    ref = np.zeros(64, dtype=np.complex64)
    for k in range(-26, 27):
        ref[(k+64)%64] = get_ltf_val(k)
    
    for dk in range(-15, 16):
        shifted = np.roll(ref, dk)
        corr = np.abs(np.sum(ref * np.conj(shifted)))
        print(f"dk={dk:3d} corr={corr:6.2f}")

if __name__ == "__main__":
    test_autocorr()
