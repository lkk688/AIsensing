#!/usr/bin/env python
import numpy as np
import collections
import time

class SDRMACLayer:
    """
    SDR MAC Layer for handling Fragmentation and Reassembly.
    
    Header Format (32 bits):
    [BlockID: 8] [FragID: 4] [LastFlag: 1] [Reserved: 19]
    """
    
    def __init__(self):
        # Buffer for reassembly: { block_id: { frag_id: payload_bits } }
        self.rx_buffer = collections.defaultdict(dict)
        # Track metadata for blocks: { block_id: {'count': N, 'timestamp': T} }
        self.rx_meta = {}
        # Maximum fragments expected (4 bits = 16 fragments max)
        self.MAX_FRAGMENTS = 16
        # TX Block ID Counter (0-255)
        self.tx_block_id = 0
        
        # Precompute power-of-2 arrays/constants if needed, but simple bit ops are fast enough.

    MAX_HEADER_BITS = 64
    MAGIC_BYTE = 0x1ACFFC1D # 32-bit Magic Word
    
    # DC Offset/Doppler 0 Protection:
    # Use 4096 bits to clear AGC transient.
    DC_PAD_BITS = 4096
    
    def _pack_header(self, block_id, frag_id, is_last):
        """
        Pack header into 64 bits + Padding.
        Format: [Pad:4096] [Magic:32] [BlockID:8] [FragID:4] [Last:1] [Pad:19]
        """
        # DC Padding (Random)
        b_dc_pad = np.tile([0, 1], self.DC_PAD_BITS // 2).astype(np.uint8)
        
        # Magic (32 bits)
        b_magic = np.array([(self.MAGIC_BYTE >> i) & 1 for i in range(31, -1, -1)], dtype=np.uint8)
        # BlockID (8 bits) - No repetition (Sequential logic handles drops)
        b_block = np.array([(block_id >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)
        
        # FragID (4 bits) - Triple Redundancy
        b_frag = np.array([(frag_id >> i) & 1 for i in range(3, -1, -1)], dtype=np.uint8)
        b_frag_3 = np.concatenate([b_frag, b_frag, b_frag]) # 12 bits
        
        # LastFlag (1 bit) - Triple Redundancy
        val_last = 1 if is_last else 0
        b_last_3 = np.array([val_last, val_last, val_last], dtype=np.uint8) # 3 bits
        
        # Reserved (9 bits)
        b_pad = np.zeros(9, dtype=np.uint8)
        
        return np.concatenate([b_dc_pad, b_magic, b_block, b_frag_3, b_last_3, b_pad])

    def validate_header(self, bits, tolerance=0):
        """
        Check if bits start with valid Magic Word (32 bits), offset by DC_PAD.
        Args:
            bits: Input bits array.
            tolerance: Max allowed bit errors (Hamming distance).
        """
        # Minimum length: Pad + Magic + Fields (> 64)
        if len(bits) < (self.DC_PAD_BITS + 32): return False
        
        offset = self.DC_PAD_BITS
        
        if np.issubdtype(bits.dtype, np.floating):
            header_bits = (bits[offset : offset+32] < 0).astype(np.uint8)
        else:
            header_bits = bits[offset : offset+32].astype(np.uint8)
            
        # Pack 32 bits -> 4 bytes -> int
        bytes_arr = np.packbits(header_bits) 
        magic = int.from_bytes(bytes_arr.tobytes(), byteorder='big')
        
        if tolerance == 0:
            return magic == self.MAGIC_BYTE
            
        # Check Hamming Distance
        xor_val = magic ^ self.MAGIC_BYTE
        diff_bits = bin(xor_val).count('1')
        return diff_bits <= tolerance

    def _unpack_header(self, bits):
        """
        Extract fields from bits (skipping padding).
        """
        if len(bits) < (self.DC_PAD_BITS + 64):
            return None, None, None
            
        offset = self.DC_PAD_BITS
        
        # Hard detection done by caller or here
        if np.issubdtype(bits.dtype, np.floating):
             bits = (bits < 0).astype(np.uint8)
             
        # Check Magic (32 bits)
        magic_bytes = np.packbits(bits[offset : offset+32]).tobytes()
        magic = int.from_bytes(magic_bytes, byteorder='big')
        
        # Tolerance check internal?
        xor_val = magic ^ self.MAGIC_BYTE
        if bin(xor_val).count('1') > 8: # Slightly relaxed internal check
             return None, None, None
            
        # BlockID (8 bits) -> bits offset+32 to offset+40
        block_id = np.packbits(bits[offset+32 : offset+40], bitorder='big')[0]
        
        # FragID (4 bits * 3) -> offset+40 to offset+52
        # FragID 1
        pad1 = np.concatenate([bits[offset+40 : offset+44], np.zeros(4, dtype=np.uint8)])
        f1 = np.packbits(pad1, bitorder='big')[0] >> 4
        # FragID 2
        pad2 = np.concatenate([bits[offset+44 : offset+48], np.zeros(4, dtype=np.uint8)])
        f2 = np.packbits(pad2, bitorder='big')[0] >> 4
        # FragID 3
        pad3 = np.concatenate([bits[offset+48 : offset+52], np.zeros(4, dtype=np.uint8)])
        f3 = np.packbits(pad3, bitorder='big')[0] >> 4
        
        # Majority Vote FragID
        if f1 == f2 or f1 == f3: frag_id = f1
        elif f2 == f3: frag_id = f2
        else: frag_id = f1
        
        # LastFlag (3 bits) -> offset+52 to offset+55
        l1 = bits[offset+52]
        l2 = bits[offset+53]
        l3 = bits[offset+54]
        last_flag = ((l1 + l2 + l3) >= 2)
        
        return int(block_id), int(frag_id), last_flag

    def fragment_packet(self, data_bits: np.ndarray, frag_size: int) -> list:
        """
        Fragment a large packet.
        
        Args:
            data_bits (np.ndarray): Binary data (1D array of 0/1).
            frag_size (int): Max size of PAYLOAD per fragment. 
                             Total size will be frag_size + 32.
                             
        Returns:
            list[np.ndarray]: List of fragment bit arrays.
        """
        total_len = len(data_bits)
        # Calculate number of fragments
        num_frags = (total_len + frag_size - 1) // frag_size
        
        if num_frags > self.MAX_FRAGMENTS:
            raise ValueError(f"Packet too large: {num_frags} fragments > Max {self.MAX_FRAGMENTS}")
            
        fragments = []
        current_bid = self.tx_block_id
        
        for i in range(num_frags):
            start = i * frag_size
            end = min(start + frag_size, total_len)
            payload = data_bits[start:end]
            
            is_last = (i == num_frags - 1)
            header = self._pack_header(current_bid, i, is_last)
            
            # Concatenate Header + Payload
            full_frag = np.concatenate([header, payload])
            fragments.append(full_frag)
            
        # Increment Block ID (wrap at 255)
        self.tx_block_id = (self.tx_block_id + 1) % 256
        
        return fragments

    def process_fragment(self, received_bits: np.ndarray):
        """
        Process a received fragment.
        """
        # Header Size = DC_PAD (32) + Start (32) + Fields (32) = 96
        # Actually MAX_HEADER_BITS = 64. So Total = 32 + 64 = 96.
        total_header_size = self.DC_PAD_BITS + 64
        
        if len(received_bits) < total_header_size:
            return None # Fragment too short
            
        header_data = received_bits[:total_header_size]
        
        # Determine if Soft LLR or Hard Bits
        if np.issubdtype(received_bits.dtype, np.floating):
            header_bits = (header_data < 0).astype(np.uint8)
        else:
            header_bits = header_data.astype(np.uint8)
            
        bid, fid, last = self._unpack_header(header_bits)
        
        if bid is None: 
            return None
            
        print(f"[MAC Debug] Processed Header: ID={bid}, Frag={fid}, Last={last}")
        
        payload = received_bits[total_header_size:]
        
        # Initialize block storage if new
        current_time = time.time()
        if bid not in self.rx_buffer:
            self.rx_buffer[bid] = {}
            self.rx_meta[bid] = {'last_received': False, 'expected_count': -1, 'ts': current_time}
            
        # Store payload (Keep original type: bits or floats)
        self.rx_buffer[bid][fid] = payload
        
        # Update metadata
        if last:
            self.rx_meta[bid]['last_received'] = True
            self.rx_meta[bid]['expected_count'] = fid + 1
            
        # Cleanup old blocks (simple garbage collection logic could go here)
        # For now, just check completion
        
        meta = self.rx_meta[bid]
        if meta['last_received']:
            count = meta['expected_count']
            # Check if we have all fragments 0..count-1
            if len(self.rx_buffer[bid]) == count:
                # Identify missing fragments? 
                # Dict size check matches expected count implies we have all unique FIDs?
                # Assuming FIDs are 0..N-1.
                # Reassemble
                sorted_frags = [self.rx_buffer[bid][i] for i in range(count) if i in self.rx_buffer[bid]]
                
                if len(sorted_frags) != count:
                    # Should not happen if len check passed
                    return None
                    
                full_packet = np.concatenate(sorted_frags)
                
                # Clear buffer
                del self.rx_buffer[bid]
                del self.rx_meta[bid]
                
                return full_packet
                
        return None

# Simple Verification
if __name__ == "__main__":
    mac = SDRMACLayer()
    
    # Create random packet (24k bits)
    packet_len = 24000
    original_packet = np.random.randint(0, 2, packet_len, dtype=np.uint8)
    
    # Fragment size 8000 (Fits in 3 frags)
    # Total frag size = 8000 + 32 header
    frags = mac.fragment_packet(original_packet, 8000)
    
    print(f"Original Packet: {len(original_packet)} bits")
    print(f"Fragments: {len(frags)}")
    for i, f in enumerate(frags):
        print(f"  Frag {i}: {len(f)} bits (Header+Payload)")
        
    # Reassembly Test (In Order)
    print("\\nProcessing In Order:")
    rx = SDRMACLayer()
    for f in frags:
        res = rx.process_fragment(f)
        if res is not None:
            print(f"  Reassembled! Length: {len(res)}")
            assert np.array_equal(res, original_packet)
            print("  Verification Successful.")
        else:
            print("  Processed fragment, waiting...")
            
    # Reassembly Test (Out of Order)
    print("\\nProcessing Out of Order:")
    rx2 = SDRMACLayer()
    rx2.process_fragment(frags[2]) # Last
    rx2.process_fragment(frags[0]) # First
    res = rx2.process_fragment(frags[1]) # Middle
    if res is not None:
        print(f"  Reassembled! Length: {len(res)}")
        assert np.array_equal(res, original_packet)
        print("  Verification Successful.")

    # LLR Polarity Test
    print("\\nLLR Polarity Test:")
    rx3 = SDRMACLayer()
    # Take Frag 0 (Bits)
    frag0 = frags[0]
    # Convert to LLR: 0 -> +10, 1 -> -10
    frag0_llr = np.where(frag0 == 0, 10.0, -10.0)
    
    # Validate Header expects: LLR < 0 -> 1.
    # Our conversion: 1 -> -10 (which is < 0). Correct.
    
    # Test validate
    is_valid = rx3.validate_header(frag0_llr)
    print(f"  Validate Header (Correct LLR): {is_valid}")
    assert is_valid
    
    # Test Inverted LLR: 0 -> -10, 1 -> +10
    frag0_llr_inv = np.where(frag0 == 0, -10.0, 10.0)
    is_valid_inv = rx3.validate_header(frag0_llr_inv)
    print(f"  Validate Header (Inverted LLR): {is_valid_inv}")
    assert not is_valid_inv
    print("  LLR Polarity Verified.")
