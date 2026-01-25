import torch
import numpy as np
import scipy.sparse as sp
import os
import csv
from enum import Enum

class BaseGraph(Enum):
    BG1 = "bg1"
    BG2 = "bg2"

class LDPC5GEncoder:
    """
    5G NR LDPC Encoder implemented in PyTorch.
    Follows 3GPP TS 38.212.
    """
    def __init__(self, k, n, num_bits_per_symbol=None, device='cuda'):
        self.k = int(k)
        self.n = int(n)
        self.num_bits_per_symbol = num_bits_per_symbol
        self.device = device
        self.coderate = k / n

        # Select Base Graph
        self.bg = self._sel_basegraph(self.k, self.coderate)
        
        # Select Lifting Factor Zc
        self.z, self.i_ls, self.k_b = self._sel_lifting(self.k, self.bg)
        
        # Load Base Graph
        self.bm = self._load_basegraph(self.i_ls, self.bg)
        
        # Calculate dimensions
        self.n_ldpc = self.bm.shape[1] * self.z
        self.k_ldpc = self.k_b * self.z
        
        # Lift Base Graph to get H
        self.pcm = self._lift_basegraph(self.bm, self.z)
        
        # Convert to PyTorch Sparse Tensor
        self.H_torch = self._sparse_scipy_to_torch(self.pcm).to(self.device)
        
        # Pre-compute encoding matrices using RU method
        self._gen_encoding_matrices()

        print(f"[LDPC] Init: k={self.k}, n={self.n}, rate={self.coderate:.2f}, BG={self.bg.value}, Z={self.z}")

    def _sel_basegraph(self, k, r):
        """Select basegraph according to 3GPP TS 38.212."""
        if k <= 292:
            return BaseGraph.BG2
        elif k <= 3824 and r <= 0.67:
            return BaseGraph.BG2
        elif r <= 0.25:
            return BaseGraph.BG2
        else:
            return BaseGraph.BG1

    def _sel_lifting(self, k, bg):
        """Select lifting size Zc and set index i_ls."""
        # Table 5.3.2-1: Sets of lifting sizes Z
        s_val = [
            [2, 4, 8, 16, 32, 64, 128, 256],
            [3, 6, 12, 24, 48, 96, 192, 384],
            [5, 10, 20, 40, 80, 160, 320],
            [7, 14, 28, 56, 112, 224],
            [9, 18, 36, 72, 144, 288],
            [11, 22, 44, 88, 176, 352],
            [13, 26, 52, 104, 208],
            [15, 30, 60, 120, 240]
        ]

        if bg == BaseGraph.BG1:
            k_b = 22
        else:
            if k > 640: k_b = 10
            elif k > 560: k_b = 9
            elif k > 192: k_b = 8
            else: k_b = 6

        # Find min Z such that k_b * Z >= k
        min_val = 100000
        z = 0
        i_ls = 0
        
        for i, s_set in enumerate(s_val):
            for s in s_set:
                x = k_b * s
                if x >= k:
                    if x < min_val:
                        min_val = x
                        z = s
                        i_ls = i
        
        # Reset k_b based on Z (though it's usually fixed for BG1)
        if bg == BaseGraph.BG1:
            k_b = 22
        else:
            k_b = 10
            
        return z, i_ls, k_b

    def _load_basegraph(self, i_ls, bg):
        """Load Base Graph from CSV."""
        # Path to codes directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "codes", f"5G_{bg.value}.csv")
        
        if bg == BaseGraph.BG1:
            bm = np.full((46, 68), -1, dtype=int)
        else:
            bm = np.full((42, 52), -1, dtype=int)
            
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)
                
                # Find start of data
                start_row = 0
                for i, row in enumerate(rows):
                    if len(row) > 0 and row[0].strip().isdigit():
                        start_row = i
                        break
                
                current_row_idx = 0
                for i in range(start_row, len(rows)):
                    row = rows[i]
                    if not row or len(row) < 3: continue
                    
                    if row[0].strip() != "":
                        current_row_idx = int(row[0])
                    
                    if row[1].strip() == "": continue
                    
                    col_idx = int(row[1])
                    if i_ls + 2 < len(row):
                        val_str = row[i_ls + 2]
                        if val_str.strip() != "":
                            bm[current_row_idx, col_idx] = int(val_str)
                            
        except Exception as e:
            print(f"[Error] Failed to load BG CSV: {e}")
            raise e
            
        return bm

    def _lift_basegraph(self, bm, z):
        """Lift basegraph to create H matrix."""
        rows, cols = bm.shape
        r_indices = []
        c_indices = []
        data = []
        
        im = np.arange(z)
        
        for r in range(rows):
            for c in range(cols):
                shift = bm[r, c]
                if shift != -1:
                    c_roll = (im + shift) % z
                    r_indices.extend(r * z + im)
                    c_indices.extend(c * z + c_roll)
                    data.extend(np.ones(z))
                    
        pcm = sp.coo_matrix((data, (r_indices, c_indices)), shape=(rows * z, cols * z))
        return pcm

    def _gen_encoding_matrices(self):
        """Generate submatrices for efficient encoding (RU method)."""
        # Based on 5G NR LDPC structure
        g = 4 # Fixed for 5G
        mb = self.bm.shape[0]
        k_b = self.k_b
        z = self.z
        
        bm_a = self.bm[0:g, 0:k_b]
        bm_b = self.bm[0:g, k_b:(k_b+g)]
        bm_c1 = self.bm[g:mb, 0:k_b]
        bm_c2 = self.bm[g:mb, k_b:(k_b+g)]
        
        # Lift submatrices
        self.pcm_a = self._lift_basegraph(bm_a, z)
        # self.pcm_b = self._lift_basegraph(bm_b, z) # Not explicit needed
        self.pcm_c1 = self._lift_basegraph(bm_c1, z)
        self.pcm_c2 = self._lift_basegraph(bm_c2, z)
        
        # Find inverse of B part
        self.pcm_b_inv = self._find_hm_b_inv(bm_b, z, self.bg)
        
        # Convert to torch sparse indices for gathering
        self.pcm_a_ind = self._mat_to_ind_torch(self.pcm_a)
        self.pcm_b_inv_ind = self._mat_to_ind_torch(self.pcm_b_inv)
        self.pcm_c1_ind = self._mat_to_ind_torch(self.pcm_c1)
        self.pcm_c2_ind = self._mat_to_ind_torch(self.pcm_c2)

    def _find_hm_b_inv(self, bm_b, z, bg):
        """Find inverse of B submatrix (dense or sparse)."""
        # See encoding.py _find_hm_b_inv logic
        # For BG1/BG2 structure, B inverse is derived from shifted identities
        
        pm_a= int(bm_b[0,0])
        if bg == BaseGraph.BG1:
            pm_b_inv = int(-bm_b[1, 0])
        else:
            pm_b_inv = int(-bm_b[2, 0])
            
        hm_b_inv = np.zeros([4*z, 4*z])
        im = np.eye(z)
        
        am = np.roll(im, pm_a, axis=1)
        b_inv = np.roll(im, pm_b_inv, axis=1)
        ab_inv = np.matmul(am, b_inv)
        
        # Row 0
        hm_b_inv[0:z, 0:z] = b_inv
        hm_b_inv[0:z, z:2*z] = b_inv
        hm_b_inv[0:z, 2*z:3*z] = b_inv
        hm_b_inv[0:z, 3*z:4*z] = b_inv
        
        # Row 1
        hm_b_inv[z:2*z, 0:z] = im + ab_inv
        hm_b_inv[z:2*z, z:2*z] = ab_inv
        hm_b_inv[z:2*z, 2*z:3*z] = ab_inv
        hm_b_inv[z:2*z, 3*z:4*z] = ab_inv
        
        # Row 2
        if bg == BaseGraph.BG1:
            hm_b_inv[2*z:3*z, 0:z] = ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = im + ab_inv
        else:
            hm_b_inv[2*z:3*z, 0:z] = im + ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = ab_inv
            
        # Row 3
        hm_b_inv[3*z:4*z, 0:z] = ab_inv
        hm_b_inv[3*z:4*z, z:2*z] = ab_inv
        hm_b_inv[3*z:4*z, 2*z:3*z] = ab_inv
        hm_b_inv[3*z:4*z, 3*z:4*z] = im + ab_inv
        
        return sp.coo_matrix(hm_b_inv)

    def _mat_to_ind_torch(self, mat):
        """Convert sparse matrix to indices for gathering."""
        mat = mat.tocoo()
        # Find max non-zeros per row
        row_counts = np.bincount(mat.row)
        max_nnz = row_counts.max()
        
        # Create dense index array [rows, max_nnz]
        # Pad with index = cols (which will point to zero in gather)
        indices = np.full((mat.shape[0], max_nnz), mat.shape[1], dtype=np.int64)
        
        # Sort by row then col
        sorted_indices = np.lexsort((mat.col, mat.row))
        r_sorted = mat.row[sorted_indices]
        c_sorted = mat.col[sorted_indices]
        
        # Fill
        # Ideally parallelize this or use efficient numpy
        counts = np.zeros(mat.shape[0], dtype=int)
        for r, c in zip(r_sorted, c_sorted):
            indices[r, counts[r]] = c
            counts[r] += 1
            
        return torch.tensor(indices, device=self.device)

    def _matmul_gather(self, mat_ind, vec):
        """
        Sparse MatMul via gather: res = M * vec
        mat_ind: [rows, max_degree], contains column indices
        vec: [batch, cols]
        """
        # Append 0 to vec for padding index
        batch_size = vec.shape[0]
        zeros = torch.zeros(batch_size, 1, device=self.device, dtype=vec.dtype)
        vec_padded = torch.cat([vec, zeros], dim=1) # [batch, cols+1]
        
        # Gather: output [batch, rows, max_degree]
        # mat_ind is [rows, max_degree]
        # We need to broadcast mat_ind to [batch, rows, max_degree]
        # or expand vec to allow gathering
        
        # PyTorch gather is strict about dims.
        # simpler: F.embedding(mat_ind, vec.T).T ?
        # Embedding: input indices [rows, max_degree], weight [cols+1, batch] -> [rows, max_degree, batch]
        # This works if batch is last dim.
        
        # Let's align dims:
        # vec_padded: [batch, cols+1]
        # mat_ind: [rows, max_degree]
        
        # Using simple loop over batch might be slow, but safe.
        # Vectorized:
        # We want res[b, r] = sum(vec[b, mat_ind[r, :]])
        
        # gathered: [batch, rows, max_degree]
        # We can use functional embedding
        # input: mat_ind [rows, max_degree]
        # weight: vec_padded.T [cols+1, batch]
        # result: [rows, max_degree, batch]
        gathered = torch.nn.functional.embedding(mat_ind, vec_padded.T)
        
        # Sum over Degree dim: [rows, batch]
        res = torch.sum(gathered, dim=1)
        
        # Transpose back to [batch, rows]
        return res.T

    def _sparse_scipy_to_torch(self, pcm):
        """Convert scipy sparse matrix to torch sparse tensor."""
        pcm = pcm.tocoo()
        values = pcm.data
        indices = np.vstack((pcm.row, pcm.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = pcm.shape
        
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()

    def encode(self, info_bits):
        """
        Encode info bits using RU method.
        info_bits: [batch, k]
        Returns: codeword [batch, n]
        """
        # Ensure input is float for matmul
        u = info_bits.float().to(self.device)
        batch_size = u.shape[0]
        
        # Add filler bits
        zeros_fill = torch.zeros(batch_size, self.k_ldpc - self.k, device=self.device)
        u_fill = torch.cat([u, zeros_fill], dim=1)
        
        # 1. p_a = B^-1 * A * u_fill
        # A * u
        Au = self._matmul_gather(self.pcm_a_ind, u_fill) % 2
        # B^-1 * (Au)
        p_a = self._matmul_gather(self.pcm_b_inv_ind, Au) % 2
        
        # 2. p_b = C1*u + C2*p_a
        p_b_1 = self._matmul_gather(self.pcm_c1_ind, u_fill) % 2
        p_b_2 = self._matmul_gather(self.pcm_c2_ind, p_a) % 2
        p_b = (p_b_1 + p_b_2) % 2
        
        # Concat [u, p_a, p_b]
        # Note: Standard order might differ, usually [ Systematic | Parity ]
        # In 5G RU method: s (systematic including filler), p_a (first 4Z parity), p_b (rest)
        c = torch.cat([u_fill, p_a, p_b], dim=1)
        
        # Rate matching / Shortening
        # Remove filler
        c_no_filler1 = c[:, :self.k]
        c_no_filler2 = c[:, self.k_ldpc:]
        c_no_filler = torch.cat([c_no_filler1, c_no_filler2], dim=1)
        
        # Puncture first 2*Z bits (usually)
        # 38.212: Output is from k to N+2Z-2 (circular buffer)
        # But we want fixed N outputs.
        # Standard implementation punctures first 2Z bits of systematic part? 
        # Actually 2*Z parity bits are punctured? 
        # "The first 2Z encoded bits are punctured" (these are part of p_a?)
        # Wait, the structure is [u p_a p_b].
        # 3GPP says first 2*Z bits of *output* are puncturing? NO.
        # "The bits are denoted by w_0, ..., w_N-1. The first 2Z bits w_0, .. w_2Z-1 are punctured."
        # w includes s and p.
        
        # For simplicity we return size N.
        # We drop the first 2*Z columns of the generated codeword?
        # Typically the first 2Z columns of H correspond to punctured variable nodes.
        # In RU method, columns are reordered.
        
        # Let's follow encoding.py logic:
        # c_short = slice(c_no_filler, [0, 2*Z], [batch, n])
        start_idx = 2 * self.z
        c_out = c_no_filler[:, start_idx : start_idx + self.n]
        
        return c_out


class LDPC5GDecoder:
    """
    Belief Propagation Decoder for 5G LDPC (Log-Domain Min-Sum).
    """
    def __init__(self, encoder: LDPC5GEncoder, max_iter=20, device='cuda'):
        self.encoder = encoder
        self.max_iter = max_iter
        self.device = device
        self.H = encoder.H_torch
        
        # Prepare edge indices for message passing
        # H is sparse [rows, cols]
        # We need adjacency list:
        # check_to_var: indices of connected variables for each check
        # var_to_check: indices of connected checks for each variable
        
        self.indices = self.H.indices() # [2, nnz]
        self.rows = self.indices[0]
        self.cols = self.indices[1]
        
        # Degrees
        self.check_degree = torch.bincount(self.rows)
        self.var_degree = torch.bincount(self.cols)
        
        # Create masks/sorters if needed for efficient gather/scatter
        # For simple pytorch implementation:
        # Messages are stored as a flat vector of size [batch, nnz]
        # We update them iteratively.
        
    def decode(self, llrs):
        """
        Decode LLRs (Log-Likelihood Ratios).
        llrs: [batch, n] -> Punctured/Shortened input.
        We need to reconstruct full codeword LLRs (size n_ldpc) including zeros for filler/punctured.
        """
        batch_size = llrs.shape[0]
        
        # Reconstruct full LLR vector layout matching H
        # Layout: [u (k), filler (k_ldpc-k), p_a, p_b]
        # Input 'llrs' corresponds to c_out from encoder, which was slice(c_no_filler, 2*Z, n)
        
        # We need to map input LLRs back to the variable nodes.
        # Unknowns (punctured) -> LLR = 0
        # Known 0 (shortened/filler) -> LLR = Infinity (strong belief in 0)
        
        # Full size excluding filler
        full_size = self.encoder.n_ldpc
        
        # Initialize full LLRs with 0 (erasure/punctured)
        L_ch = torch.zeros(batch_size, full_size, device=self.device)
        
        # Indices in full array
        # Input starts at 2*Z of (c without filler)
        # We need to handle filler insertion.
        # c is [u, filler, p_a, p_b]
        # c_no_filler is [u, p_a, p_b]
        # Input is c_no_filler[2*Z : 2*Z + n]
        
        # Map input LLRs to c_no_filler mapping
        # c_no_filler indices:
        # 0..k-1 -> u
        # k..k+len(p_a+p_b)-1 -> parity
        
        # 2*Z typically falls within u (if k > 2Z) or into p_a?
        # Z is small (e.g., 384). 2*Z = 768.
        # If k=8000, 2*Z is inside u.
        # So first 2*Z systematic bits are punctured? Yes, usually.
        
        # Logic:
        # valid_indices in c_no_filler: range(2*Z, 2*Z + n)
        # We map these to L_ch.
        # But L_ch must include filler.
        
        # Construct L_ch for [u, filler, p_a, p_b]
        # 1. u:
        #    - first 2*Z bits of u are punctured (0)
        #    - n bits follow.
        #    - BUT u might be split.
        
        # Let's re-trace encoder slices:
        # c = [u, filler, p_a, p_b]
        # c_no_filler = [u, p_a, p_b] (filler removed)
        # output = c_no_filler[2*Z : 2*Z + n]
        
        # So we simply reverse this.
        # Create temp buffer matching c_no_filler size
        c_no_f_len = self.encoder.k + (self.encoder.n_ldpc - self.encoder.k_ldpc) # u + p
        temp_L = torch.zeros(batch_size, c_no_f_len, device=self.device)
        
        # Fill known LLRs
        start = 2 * self.encoder.z
        end = start + self.encoder.n
        # Clip end if larger than buffer (shouldn't be if sizes match)
        if end > c_no_f_len:
             # This happens if rate is very low or N is large?
             # Just map what we have.
             # 5G rate matching repeats bits if N > N_circular_buffer.
             # But here we assume N fit in.
             pass
             
        # Copy input LLRs to corresponding positions
        # Indices relative to c_no_filler
        temp_L[:, start:end] = llrs
        
        # Now insert "infinity" for filler bits
        # filler is at index k in c (between u and p_a)
        # c_no_filler has u (0..k) and p (k..end)
        # We need to expand to [u (0..k), filler, p (k..end)]
        
        # Infinity belief for 0: very large positive LLR
        INF = 1e9
        filler = torch.full((batch_size, self.encoder.k_ldpc - self.encoder.k), INF, device=self.device)
        
        u_part = temp_L[:, :self.encoder.k]
        p_part = temp_L[:, self.encoder.k:]
        
        L_prior = torch.cat([u_part, filler, p_part], dim=1)
        
        # Run Min-Sum Algorithm
        # Messages: [batch, nnz]
        # We iterate:
        # 1. Variable to Check: M_{v->c} = L_prior[v] + sum(M_{c'->v}) - M_{c->v}
        #    Initially M_{c->v} = 0, so M_{v->c} = L_prior[v]
        # 2. Check to Variable: M_{c->v} = prod(sign) * min(|M_{v->c}|)
        
        # Optimized tensor ops:
        # Flatten L_prior to map to edges using 'cols' index
        # messages_vc: [batch, nnz]
        # messages_cv: [batch, nnz], init 0
        
        nnz = self.rows.shape[0]
        messages_cv = torch.zeros(batch_size, nnz, device=self.device)
        
        for it in range(self.max_iter):
            # 1. Variable Node Update
            # Get full L_v (intrinsic + extrinsic)
            # Extrinsic sum: scatter_add messages_cv to variables
            extrinsic = torch.zeros_like(L_prior)
            # scatter_add(src, dim, index, out) -> index must be broadcasted?
            # PyTorch scatter_add_ expects index to match src dims
            # Expanded index: [batch, nnz]
            idx_cols = self.cols.expand(batch_size, -1)
            extrinsic.scatter_add_(1, idx_cols, messages_cv)
            
            L_posterior = L_prior + extrinsic
            
            # M_vc = L_posterior[v] - M_cv
            # Gather L_posterior for each edge
            L_v_gather = torch.gather(L_posterior, 1, idx_cols)
            messages_vc = L_v_gather - messages_cv
            
            # 2. Check Node Update (Min-Sum)
            # M_cv = prod(sign(M_vc)) * min(|M_vc|)
            # Group by check node 'rows'
            # This is tricky in pure PyTorch without scatter_min/max
            # "scatter_reduce" with "min" (available in recent PyTorch)
            # or torch_scatter package (prefer pure torch)
            
            # Min-Sum approx with Tanh rule
            # 2 * atanh( prod( tanh(M_vc / 2) ) )
            # Use log-domain trick to avoid underflow?
            # Or standard Min-Sum with masking
            
            # Tanh implementation is easier to vectorize fully
            # limit messages to avoid nan
            messages_vc = torch.clamp(messages_vc, -20, 20)
            tanh_msg = torch.tanh(messages_vc / 2.0)
            
            # Product over check nodes
            # We need to compute product of incoming edges for each check node
            # and then exclude self.
            # dense approach: create [batch, num_checks] array?
            # Sparse reduction: 
            # We can't easily do "product" scatter_reduce in older pytorch.
            # Log-trick: sum(ln(|tanh|)) * prod(sign)
            
            # Let's assume standard Min-Sum
            # Compute sign product and min magnitude
            # This is hard to do cleanly without scatter_min/prod.
            
            # Alternative: Tanh Product using scatter_add (sum of logs)
            # P = prod(t) => log(|P|) = sum(log(|t|))
            # sign(P) = prod(sign(t)) -> sum( (1-sign)/2 ) mod 2 ?
            
            abs_tanh = torch.abs(tanh_msg)
            sign_tanh = torch.sign(tanh_msg)
            # fix zero values
            abs_tanh = torch.clamp(abs_tanh, min=1e-12)
            
            log_abs_tanh = torch.log(abs_tanh)
            
            # Scatter sum these to check nodes
            idx_rows = self.rows.expand(batch_size, -1)
            
            sum_log = torch.zeros(batch_size, self.encoder.H_torch.shape[0], device=self.device)
            sum_log.scatter_add_(1, idx_rows, log_abs_tanh)
            
            # For signs: count negatives
            is_neg = (sign_tanh < 0).float()
            sum_neg = torch.zeros_like(sum_log)
            sum_neg.scatter_add_(1, idx_rows, is_neg)
            
            # Total Check Logic
            total_log = torch.gather(sum_log, 1, idx_rows)
            total_neg = torch.gather(sum_neg, 1, idx_rows)
            
            # Subtract self to get extrinsic M_cv
            ext_log = total_log - log_abs_tanh
            ext_neg_count = total_neg - is_neg
            
            # Reconstruct
            # sign = (-1)^ext_neg_count
            ext_sign = torch.where(ext_neg_count % 2 == 0, 1.0, -1.0)
            ext_val = torch.exp(ext_log)
            
            # Back to LLR domain: 2 * atanh( sign * val )
            # clamp val to avoid 1.0 (inf)
            ext_val = torch.clamp(ext_val, max=0.999999)
            messages_cv = 2.0 * torch.atanh(ext_sign * ext_val)
            
        # Final Decision
        # Extrinsic sum updated one last time
        extrinsic.zero_()
        extrinsic.scatter_add_(1, idx_cols, messages_cv)
        L_final = L_prior + extrinsic
        
        # Extract systematic bits (u)
        # u is first k bits
        L_u = L_final[:, :self.encoder.k]
        
        # Hard decision
        bits = (L_u < 0).float() # LLR definition dependent: usually L = log(P(0)/P(1)). If < 0 -> 1.
        
        return bits

