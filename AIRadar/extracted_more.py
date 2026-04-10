class FiLMConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, cond_dim, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.film = FiLMLayer(out_ch, cond_dim)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return F.relu(x)

class SEBlock(nn.Module):

    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ComplexConvBlock(nn.Module):
    """
    Complex convolution that respects I/Q phase-amplitude coupling.
    Mathematically: (A+Bi)*(C+Di) = (AC-BD) + (AD+BC)i
    Uses GroupNorm instead of BatchNorm for stability in signal processing.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm_re = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm_im = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x_re, x_im):
        out_re = self.conv_re(x_re) - self.conv_im(x_im)
        out_im = self.conv_re(x_im) + self.conv_im(x_re)
        out_re = F.relu(self.norm_re(out_re))
        out_im = F.relu(self.norm_im(out_im))
        return (out_re, out_im)