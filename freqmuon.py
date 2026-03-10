# freqmuon.py
#
# "No compromise" per-step full frequency-bin orthogonalization on an MxM circular grid,
# but optimized to reduce type casting and allocations:
# - uses rfft2/irfft2 (exact for real inputs) to avoid full FFT + symmetry fills
# - avoids realification via cat() (2m x 2n). Instead does NS5 on split (Re, Im) directly
# - keeps momentum buffers in float32 for conv filters (only one grad->float32 cast)
# - batches conv params with the same (Cout,Cin,kH,kW) shape
#
# Usage:
#   python freqmuon.py --runs 1 --seed 0 --fft_size 8 --ns_steps 2

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import airbench94_muon as ab


@dataclass
class FreqMuonCfg:
    fft_size: int = 8
    ns_steps: int = 2
    eps: float = 1e-7


def _zeropower_ns5_complex(
    X: torch.Tensor,  # [B, m, n], complex64
    *,
    steps: int,
    eps: float,
) -> torch.Tensor:
    """
    Muon-style NS5 polynomial on native complex tensors.
    Returns approx polar factor / orthogonalized update.
    """
    assert X.ndim == 3 and X.is_complex()
    _, m, n = X.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    norm = torch.linalg.matrix_norm(X, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / norm
    if transpose := m > n:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.mH      # [B, r, r]
        X = a * X + (b * A + c * A @ A) @ X
    if transpose:
        X = X.transpose(-2, -1)
    return X


# _zeropower_ns5_complex = torch.compile(_zeropower_ns5_complex, mode="max-autotune")


def _freq_muon_conv_update_batched(g32: torch.Tensor, cfg: FreqMuonCfg) -> torch.Tensor:
    """
    g32: [P, Cout, Cin, kH, kW] float32
    returns: [P, Cout, Cin, kH, kW] float32
    """
    assert g32.ndim == 5 and g32.dtype == torch.float32
    P, Cout, Cin, kH, kW = g32.shape
    M = cfg.fft_size

    if kH > M or kW > M:
        raise ValueError(f"Kernel {kH}x{kW} > fft_size {M}. Increase --fft_size.")

    # Full FFT on an MxM logical grid; PyTorch pads/trims automatically.
    Khat = torch.fft.fft2(g32, s=(M, M), dim=(-2, -1), norm="ortho")  # [P, Cout, Cin, M, M]

    Kflat = Khat.permute(0, 3, 4, 1, 2).reshape(P * M * M, Cout, Cin)
    Kflat = _zeropower_ns5_complex(Kflat, steps=cfg.ns_steps, eps=cfg.eps)
    Khat2 = Kflat.reshape(P, M, M, Cout, Cin).permute(0, 3, 4, 1, 2)

    # Back to spatial domain, then crop to original kernel support
    upd_pad = torch.fft.ifft2(Khat2, s=(M, M), dim=(-2, -1), norm="ortho")
    upd = upd_pad.real[:, :, :, :kH, :kW]
    return upd.to(dtype=g32.dtype)


_freq_muon_conv_update_batched = torch.compile(_freq_muon_conv_update_batched, mode="max-autotune")


class MuonFreqUltraFast(torch.optim.Optimizer):
    """
    Muon-like optimizer for conv filters, but using full-bin frequency-domain polar updates.
    Optimized to avoid unnecessary casting and allocations.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, cfg: FreqMuonCfg = FreqMuonCfg()):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov requires momentum > 0")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.cfg = cfg

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            # Bucket conv params by shape to batch FFT + polar
            buckets: Dict[Tuple[int, int, int, int, torch.dtype, torch.device], List[torch.Tensor]] = {}
            ge32_map: Dict[torch.Tensor, torch.Tensor] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                st = self.state[p]
                if "momentum_buffer32" not in st:
                    # Float32 momentum buffer to eliminate repeated dtype churn
                    st["momentum_buffer32"] = torch.zeros_like(g, dtype=torch.float32)
                    st["wn_scale"] = math.sqrt(p.data.numel())

                buf32 = st["momentum_buffer32"]
                buf32.mul_(momentum).add_(g.to(torch.float32))
                ge32 = buf32 if not nesterov else (buf32 + momentum * g.to(torch.float32))
                ge32_map[p] = ge32

                if p.data.ndim != 4:
                    # Fallback: simple SGD for non-conv params in this optimizer
                    p.data.add_(ge32.to(p.data.dtype), alpha=-lr)
                    continue

                Cout, Cin, kH, kW = p.data.shape
                key = (Cout, Cin, kH, kW, p.data.dtype, p.data.device)
                buckets.setdefault(key, []).append(p)

            for (Cout, Cin, kH, kW, dt, dev), plist in buckets.items():
                # Keep airbench-style weight normalization (in param dtype, as baseline does)
                for p in plist:
                    st = self.state[p]
                    scale = st["wn_scale"]
                    p.data.mul_(scale / (p.data.norm() + 1e-12))

                ge_stack = torch.stack([ge32_map[p] for p in plist], dim=0)  # float32 [P,Cout,Cin,kH,kW]

                upd32 = _freq_muon_conv_update_batched(ge_stack, self.cfg)    # float32

                # Apply update (single cast here)
                upd = upd32.to(dt)
                for i, p in enumerate(plist):
                    p.data.add_(upd[i], alpha=-lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--fft_size", type=int, default=8)
    parser.add_argument("--ns_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = FreqMuonCfg(fft_size=args.fft_size, ns_steps=args.ns_steps)

    # Patch airbench Muon factory
    def _make_muon(params, lr=1e-3, momentum=0.0, nesterov=False):
        return MuonFreqUltraFast(params, lr=lr, momentum=momentum, nesterov=nesterov, cfg=cfg)

    ab.Muon = _make_muon

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if not args.no_compile:
        model.compile(mode="default")

    ab.print_columns(ab.logging_columns_list, is_head=True)
    
    def run_fn(run, model):
        torch.manual_seed(args.seed + run if run != "warmup" else args.seed - 1)
        return ab.main(run, model)

    # run_fn("warmup", model)

    accs = torch.tensor([run_fn(run, model) for run in range(args.runs)])
    print("Mean: %.4f Std: %.4f" % (accs.mean(), accs.std()))


if __name__ == "__main__":
    main()
