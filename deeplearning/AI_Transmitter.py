


class Transmitter():
    def __init__(self, channeldataset='deepmimo', channeltype="ofdm", scenario='O1_60', dataset_folder='data/DeepMIMO', \
                 direction="uplink", num_ut = 1, num_ut_ant=2, num_bs = 1, num_bs_ant=16,\
                 batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                 subcarrier_spacing=15e3, num_guard_carriers=None, pilot_ofdm_symbol_indices=None, \
                USE_LDPC = True, pilot_pattern = "kronecker", guards = True, showfig = True, savedata=True, outputpath=None,
                USE_NN_RECEIVER = False) -> None:
                #num_guard_carriers=[15,16]
        self.channeltype = channeltype
        self.channeldataset = channeldataset
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.num_ofdm_symbols = num_ofdm_symbols
        self.num_bits_per_symbol = num_bits_per_symbol
        self.showfig = showfig
        self.savedata = savedata
        self.pilot_pattern = pilot_pattern
        self.scenario = scenario
        self.dataset_folder = dataset_folder
        self.direction = direction
        self.num_ut = num_ut #num_rx #1
        self.num_bs = num_bs #num_tx #1
        self.num_ut_ant = num_ut_ant #num_rx #2 #4
        self.num_bs_ant = num_bs_ant #8
        self.num_time_steps = 1 #num_ofdm_symbols #??? 
        self.outputpath = outputpath
        self.USE_NN_RECEIVER = USE_NN_RECEIVER
        # ... existing code ...
        
        # Initialize neural network receiver if enabled
        if self.USE_NN_RECEIVER:
            self.init_nn_receiver()
            
    # ... existing code ...
    
    def init_nn_receiver(self):
        """Initialize the neural network receiver components"""
        # Define the neural network architecture for channel estimation
        self.nn_channel_estimator = NNChannelEstimator(
            num_rx_ant=self.num_bs_ant,
            num_tx_ant=self.num_ut_ant,
            num_ofdm_symbols=self.num_ofdm_symbols,
            num_subcarriers=self.RESOURCE_GRID.num_effective_subcarriers
        )
        
        # Define the neural network architecture for equalization
        self.nn_equalizer = NNEqualizer(
            num_rx_ant=self.num_bs_ant,
            num_tx_ant=self.num_ut_ant,
            num_streams=self.num_streams_per_tx,
            num_ofdm_symbols=self.num_ofdm_symbols,
            num_subcarriers=self.RESOURCE_GRID.num_effective_subcarriers
        )
        
        # Define the neural network architecture for demapping
        self.nn_demapper = NNDemapper(
            num_bits_per_symbol=self.num_bits_per_symbol,
            num_streams=self.num_streams_per_tx
        )
        
        # Optimizer for training
        self.optimizer = torch.optim.Adam([
            {'params': self.nn_channel_estimator.parameters()},
            {'params': self.nn_equalizer.parameters()},
            {'params': self.nn_demapper.parameters()}
        ], lr=0.001)
        
        # Loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def train_nn_receiver(self, num_epochs=10, datapath="data/training_data.npy"):
        """Train the neural network receiver using saved data"""
        if not self.USE_NN_RECEIVER:
            print("Neural network receiver is not enabled.")
            return
            
        # Load training data
        try:
            data = np.load(datapath, allow_pickle=True).item()
            y = torch.tensor(data['y'], dtype=torch.complex64)
            b = torch.tensor(data['b'], dtype=torch.float32)
            no = torch.tensor(data['no'], dtype=torch.float32)
        except:
            print(f"Could not load training data from {datapath}")
            return
            
        # Training loop
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass through the neural network receiver
            b_hat_logits = self.nn_receiver_forward(y, no)
            
            # Compute loss
            loss = self.criterion(b_hat_logits, b)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if epoch % 10 == 0:
                with torch.no_grad():
                    b_hat = (b_hat_logits > 0).float()
                    ber = torch.mean((b_hat != b).float())
                print(f"Epoch {epoch}, Loss: {loss.item()}, BER: {ber.item()}")
                
        # Save trained model
        torch.save({
            'channel_estimator': self.nn_channel_estimator.state_dict(),
            'equalizer': self.nn_equalizer.state_dict(),
            'demapper': self.nn_demapper.state_dict()
        }, "data/nn_receiver_model.pt")
        
    def nn_receiver_forward(self, y, no):
        """Forward pass through the neural network receiver"""
        # Convert inputs to PyTorch tensors if they're not already
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.complex64)
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, dtype=torch.float32)
            
        # Channel estimation
        h_hat = self.nn_channel_estimator(y, no)
        
        # Equalization
        x_hat = self.nn_equalizer(y, h_hat, no)
        
        # Demapping
        llr = self.nn_demapper(x_hat, no)
        
        return llr
        
    def nn_receiver(self, y, no, b=None):
        """Neural network based receiver implementation"""
        if not self.USE_NN_RECEIVER:
            print("Neural network receiver is not enabled. Using conventional receiver.")
            return self.receiver(y, no, None, b)
            
        # Forward pass through the neural network receiver
        llr_est = self.nn_receiver_forward(y, no)
        
        # Convert to numpy for consistency with the rest of the code
        llr_est = llr_est.detach().numpy()
        
        # Decision
        if self.USE_LDPC:
            b_hat_tf = self.decoder(llr_est)
            b_hat = b_hat_tf.numpy()
        else:
            b_hat = hard_decisions(llr_est, np.int32)
            
        # Calculate BER if ground truth is provided
        BER = None
        if b is not None:
            BER = calculate_BER(b, b_hat)
            print("NN Receiver BER Value:", BER)
            
        return b_hat, BER
    
    def __call__(self, b=None, ebno_db=15.0, perfect_csi=False, datapath="data/saved_data.npy", use_nn_receiver=None):
        # Allow overriding the USE_NN_RECEIVER setting for this call
        use_nn = self.USE_NN_RECEIVER if use_nn_receiver is None else use_nn_receiver
        
        # Compute noise power
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        no = np.float32(no)
        
        # Generate channel
        h_b, tau_b = self.get_channelcir()
        if self.channeltype == 'ofdm':
            h_out = self.get_OFDMchannelresponse(h_b, tau_b)
        elif self.channeltype == 'time':
            h_out = self.get_timechannelresponse(h_b, tau_b)
            
        # Transmitter
        y, x_rg, x, b = self.uplinktransmission(b=b, no=no, h_out=h_out)
        
        # Receiver processing
        if use_nn:
            # Use neural network receiver
            b_hat, BER = self.nn_receiver(y, no, b)
        else:
            # Use conventional receiver
            x_hat, no_eff, h_hat, err_var, h_perfect, err_var_perfect = self.channelest_equ(
                y, no, h_b=h_b, tau_b=tau_b, h_out=h_out, perfect_csi=perfect_csi
            )
            b_hat, llr_est = self.demapper_decision(x_hat=x_hat, no_eff=no_eff)
            BER = calculate_BER(b, b_hat)
            print("Conventional BER Value:", BER)
            
        # Save data if required
        if self.savedata:
            saved_data = self.save_parameters()
            saved_data['no'] = no
            saved_data['h_b'] = h_b
            saved_data['tau_b'] = tau_b
            saved_data['h_out'] = h_out
            saved_data['y'] = y
            saved_data['x_rg'] = x_rg
            saved_data['x'] = x
            saved_data['b'] = b
            saved_data['b_hat'] = b_hat
            saved_data['BER'] = BER
            saved_data['use_nn_receiver'] = use_nn
            
            if not use_nn:
                saved_data['x_hat'] = x_hat
                saved_data['no_eff'] = no_eff
                saved_data['h_hat'] = to_numpy(h_hat)
                saved_data['err_var'] = err_var
                saved_data['h_perfect'] = h_perfect
                saved_data['err_var_perfect'] = err_var_perfect
                saved_data['llr_est'] = llr_est
                
            np.save(datapath, saved_data)
            
        return b_hat, BER

# Neural Network Models for the Receiver
class NNChannelEstimator(torch.nn.Module):
    def __init__(self, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers):
        super().__init__()
        
        # Input features: received signal and noise level
        input_dim = 2 * num_ofdm_symbols * num_subcarriers  # Real and imaginary parts
        
        # Output features: channel estimates for all TX-RX antenna pairs
        output_dim = 2 * num_rx_ant * num_tx_ant * num_ofdm_symbols * num_subcarriers  # Real and imaginary parts
        
        # Define neural network layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim)
        )
        
    def forward(self, y, no):
        # Flatten and concatenate real and imaginary parts
        batch_size = y.shape[0]
        y_flat = y.view(batch_size, -1)
        y_real = torch.real(y_flat)
        y_imag = torch.imag(y_flat)
        y_features = torch.cat([y_real, y_imag], dim=1)
        
        # Add noise level as a feature
        no_expanded = no.expand(batch_size, 1)
        features = torch.cat([y_features, no_expanded], dim=1)
        
        # Forward pass
        h_flat = self.layers(features)
        
        # Reshape to complex tensor
        h_real, h_imag = torch.chunk(h_flat, 2, dim=1)
        h_complex = torch.complex(h_real, h_imag)
        
        # Reshape to expected output dimensions
        h_hat = h_complex.view(batch_size, -1, 1, -1, 1, -1, -1)  # Match conventional estimator output shape
        
        return h_hat

class NNEqualizer(torch.nn.Module):
    def __init__(self, num_rx_ant, num_tx_ant, num_streams, num_ofdm_symbols, num_subcarriers):
        super().__init__()
        
        # Input features: received signal, channel estimates, and noise level
        input_dim = 2 * (num_ofdm_symbols * num_subcarriers + 
                         num_rx_ant * num_tx_ant * num_ofdm_symbols * num_subcarriers)  # Real and imaginary parts
        
        # Output features: equalized symbols
        output_dim = 2 * num_streams * num_ofdm_symbols * num_subcarriers  # Real and imaginary parts
        
        # Define neural network layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim)
        )
        
    def forward(self, y, h_hat, no):
        # Flatten and concatenate real and imaginary parts
        batch_size = y.shape[0]
        y_flat = y.view(batch_size, -1)
        y_real = torch.real(y_flat)
        y_imag = torch.imag(y_flat)
        
        h_flat = h_hat.view(batch_size, -1)
        h_real = torch.real(h_flat)
        h_imag = torch.imag(h_flat)
        
        # Concatenate features
        features = torch.cat([y_real, y_imag, h_real, h_imag], dim=1)
        
        # Add noise level as a feature
        no_expanded = no.expand(batch_size, 1)
        features = torch.cat([features, no_expanded], dim=1)
        
        # Forward pass
        x_flat = self.layers(features)
        
        # Reshape to complex tensor
        x_real, x_imag = torch.chunk(x_flat, 2, dim=1)
        x_complex = torch.complex(x_real, x_imag)
        
        # Reshape to expected output dimensions
        x_hat = x_complex.view(batch_size, -1, -1, -1)  # Match conventional equalizer output shape
        
        return x_hat

class NNDemapper(torch.nn.Module):
    def __init__(self, num_bits_per_symbol, num_streams):
        super().__init__()
        
        # Input features: equalized symbols
        input_dim = 2 * num_streams  # Real and imaginary parts per symbol
        
        # Output features: LLRs for each bit
        output_dim = num_bits_per_