import math
import os
from functools import reduce

import matplotlib
import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from scipy.special import betainc
from scipy.stats import norm, truncnorm
from tqdm import tqdm

from utils.rid_utils import (
    RADIUS,
    RADIUS_CUTOFF,
    fft,
    generate_Fourier_watermark_latents,
    get_distance,
    ifft,
    make_Fourier_ringid_pattern,
    ring_mask,
)

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_subplots(*args, **kwargs):
    try:
        return plt.subplots(*args, **kwargs)
    except Exception:
        plt.switch_backend("Agg")
        return plt.subplots(*args, **kwargs)


class ConfigurableHybridWatermarker:
    def __init__(
        self,
        device,
        gs_ch_factor=1,
        gs_hw_factor=1,
        ring_radius=RADIUS,
        ring_radius_cutoff=RADIUS_CUTOFF,
        fpr_target=1e-6,
        user_number=1,
        user_id=0,
        gs_channels=None,
        ring_channels=None,
        ring_value_range=32,
        ring_threshold_dist=None,
        ring_threshold_base=30.8,
        ring_threshold_base_channels=2,
        ring_threshold_base_value_range=32,
        debug=False,
        use_chacha=False,
        chacha_seed=None,
    ):
        self.device = device
        self.debug = debug
        self.use_chacha = use_chacha
        self.chacha_seed = chacha_seed

        self.latent_shape = (4, 64, 64)
        self.gs_ch = gs_ch_factor
        self.gs_hw = gs_hw_factor
        self.gs_channels = gs_channels if gs_channels is not None else [0, 3]
        self.ring_channels = ring_channels if ring_channels is not None else [1, 2]

        self._validate_channels()

        self.gs_mark_length = (
            len(self.gs_channels) * 64 * 64
        ) // (self.gs_ch * self.gs_hw * self.gs_hw)
        self.gs_threshold_vote = (
            1
            if self.gs_hw == 1 and self.gs_ch == 1
            else self.gs_ch * self.gs_hw * self.gs_hw // 2
        )

        self.ring_radius = ring_radius
        self.ring_radius_cutoff = ring_radius_cutoff
        self.ring_mark_length = ring_radius - ring_radius_cutoff
        self.ring_value_range = ring_value_range

        self.ring_mask_single = torch.tensor(
            ring_mask(size=64, r_out=self.ring_radius, r_in=self.ring_radius_cutoff)
        ).to(self.device)
        self.ring_masks = torch.stack(
            [self.ring_mask_single for _ in self.ring_channels]
        ).to(self.device)

        self.user_number = user_number
        self.tau_gs = self._calculate_bit_threshold(self.gs_mark_length, fpr_target, 1)
        self.tau_gs_traceable = self._calculate_bit_threshold(
            self.gs_mark_length, fpr_target, user_number
        )

        self.ring_threshold_dist = (
            ring_threshold_dist
            if ring_threshold_dist is not None
            else self._auto_ring_threshold(
                ring_threshold_base,
                ring_threshold_base_channels,
                ring_threshold_base_value_range,
            )
        )

        self.tp_detection_count = 0
        self.tp_traceability_count = 0

        self.gs_key = None
        self.gs_nonce = None
        self.gs_msg = None
        self.ring_message = None
        self.ring_key_values = None

        self._gs_key_rng = None
        if self.use_chacha and self.chacha_seed is not None:
            self._gs_key_rng = torch.Generator(device="cpu").manual_seed(
                int(self.chacha_seed)
            )

        self.ring_candidate_patterns, self.ring_user_ids = self._generate_candidate_database(
            self.user_number
        )
        self.ring_current_user_id = user_id
        self.output_latents = None

    def _validate_channels(self):
        if len(self.gs_channels) == 0:
            raise ValueError("gs_channels must be non-empty.")
        if len(self.ring_channels) == 0:
            raise ValueError("ring_channels must be non-empty.")
        if len(set(self.gs_channels) & set(self.ring_channels)) > 0:
            raise ValueError("gs_channels and ring_channels must be disjoint.")
        if len(self.gs_channels) % self.gs_ch != 0:
            raise ValueError("len(gs_channels) must be divisible by gs_ch_factor.")
        max_channel = self.latent_shape[0] - 1
        for ch in self.gs_channels + self.ring_channels:
            if ch < 0 or ch > max_channel:
                raise ValueError(f"channel {ch} is out of range 0..{max_channel}")

    def _auto_ring_threshold(
        self, base_threshold, base_channels, base_value_range
    ):
        value_scale = self.ring_value_range / float(base_value_range)
        channel_scale = float(base_channels) / len(self.ring_channels)
        return base_threshold * value_scale * channel_scale

    def _stream_key_encrypt(self, bits: np.ndarray):
        if not self.use_chacha:
            return bits
        if self._gs_key_rng is None:
            self.gs_key = get_random_bytes(32)
            self.gs_nonce = get_random_bytes(12)
        else:
            key = (
                torch.randint(
                    0, 256, (32,), generator=self._gs_key_rng, dtype=torch.uint8
                )
                .numpy()
                .tobytes()
            )
            nonce = (
                torch.randint(
                    0, 256, (12,), generator=self._gs_key_rng, dtype=torch.uint8
                )
                .numpy()
                .tobytes()
            )
            self.gs_key = key
            self.gs_nonce = nonce
        cipher = ChaCha20.new(key=self.gs_key, nonce=self.gs_nonce)
        m_byte = cipher.encrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return m_bit

    def _stream_key_decrypt(self, bits: np.ndarray):
        if not self.use_chacha:
            return bits
        cipher = ChaCha20.new(key=self.gs_key, nonce=self.gs_nonce)
        m_byte = cipher.decrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return m_bit

    def _derive_ring_key_nonce(self, user_id: int):
        gen = torch.Generator(device="cpu").manual_seed(int(user_id))
        key = (
            torch.randint(0, 256, (32,), generator=gen, dtype=torch.uint8)
            .numpy()
            .tobytes()
        )
        nonce = (
            torch.randint(0, 256, (12,), generator=gen, dtype=torch.uint8)
            .numpy()
            .tobytes()
        )
        return key, nonce

    def _ring_encrypt_message(self, user_id: int, message: torch.Tensor):
        if not self.use_chacha:
            return message
        bits = message.flatten().detach().cpu().numpy().astype(np.uint8)
        key, nonce = self._derive_ring_key_nonce(user_id)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        m_byte = cipher.encrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return torch.from_numpy(m_bit).to(self.device).reshape(message.shape)

    def _calculate_bit_threshold(self, length, fpr, user_num):
        for i in range(length):
            p_val = betainc(i + 1, length - i, 0.5) * user_num
            if p_val <= fpr:
                return i / length
        return 0.8

    def _trunc_sampling(self, message):
        m_flat = message.flatten()
        m_bin = (m_flat > 0.5).astype(np.int64)
        z = np.zeros_like(m_bin, dtype=np.float32)
        ppf = [norm.ppf(0.0), norm.ppf(0.5), norm.ppf(1.0)]
        for i, val in enumerate(m_bin):
            z[i] = truncnorm.rvs(ppf[int(val)], ppf[int(val) + 1])
        return (
            torch.from_numpy(z)
            .reshape(len(self.gs_channels), 64, 64)
            .to(self.device)
            .half()
        )

    def visualize_hybrid_latents(self, output_latents, save_path="hybrid_debug.png"):
        latents = output_latents[0].detach().cpu()
        total = len(self.ring_channels) + len(self.gs_channels)
        cols = min(4, total)
        rows = int(math.ceil(total / float(cols)))
        fig, axes = _safe_subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.atleast_1d(axes).reshape(-1)

        # ring channels (freq)
        for i, ch in enumerate(self.ring_channels):
            ch_fft = fft(output_latents[:, ch : ch + 1])[0, 0].real.detach().cpu().numpy()
            ch_fft = np.nan_to_num(ch_fft, nan=0.0, posinf=0.0, neginf=0.0)
            im = axes[i].imshow(ch_fft, cmap="RdBu_r", vmin=-64, vmax=64)
            axes[i].set_title(f"Channel {ch} FFT (Real)")
            fig.colorbar(im, ax=axes[i])

        # gs channels (spatial)
        offset = len(self.ring_channels)
        for i, ch in enumerate(self.gs_channels):
            ch_spatial = latents[ch].numpy()
            ch_spatial = np.nan_to_num(ch_spatial, nan=0.0, posinf=0.0, neginf=0.0)
            im = axes[offset + i].imshow(ch_spatial, cmap="viridis")
            axes[offset + i].set_title(f"Channel {ch} Spatial (GS)")
            fig.colorbar(im, ax=axes[offset + i])

        for j in range(total, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Hybrid Watermark (Dynamic Channels)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _generate_candidate_database(self, num_users):
        candidate_patterns = []
        user_ids = []
        for user_id in tqdm(range(num_users)):
            torch.manual_seed(user_id)
            user_ring_message = torch.randint(
                0, 2, (self.ring_mark_length, len(self.ring_channels))
            ).to(self.device)
            user_ring_message = self._ring_encrypt_message(user_id, user_ring_message)
            user_key_values = torch.where(
                user_ring_message == 1, self.ring_value_range, -self.ring_value_range
            )
            pattern = make_Fourier_ringid_pattern(
                device=self.device,
                key_value_combination=user_key_values,
                no_watermark_latents=torch.zeros(1, *self.latent_shape).to(self.device),
                radius=self.ring_radius,
                radius_cutoff=self.ring_radius_cutoff,
                ring_watermark_channel=self.ring_channels,
                heter_watermark_channel=[],
            )
            candidate_patterns.append(pattern)
            user_ids.append(user_id)

        for pattern in candidate_patterns:
            pattern[:, self.ring_channels, ...] = fft(
                torch.fft.fftshift(
                    ifft(pattern[:, self.ring_channels, ...]), dim=(-1, -2)
                )
            )
        return candidate_patterns, user_ids

    def create_watermark_and_return_w(self, base_latents, user_id):
        output_latents = base_latents.clone()
        self.ring_current_user_id = user_id

        self.gs_msg = torch.randint(
            0,
            2,
            (
                len(self.gs_channels) // self.gs_ch,
                64 // self.gs_hw,
                64 // self.gs_hw,
            ),
            device=self.device,
        )
        sd = self.gs_msg.repeat(1, self.gs_ch, self.gs_hw, self.gs_hw)
        if self.use_chacha:
            m_bits = self._stream_key_encrypt(sd.flatten().cpu().numpy())
            m = m_bits.reshape(len(self.gs_channels), 64, 64)
        else:
            self.gs_key = torch.randint(
                0, 2, (len(self.gs_channels), 64, 64), device=self.device
            )
            m = ((sd + self.gs_key) % 2).cpu().numpy()
        gs_w = self._trunc_sampling(m)

        for i, ch in enumerate(self.gs_channels):
            output_latents[:, ch] = gs_w[i]

        output_latents = generate_Fourier_watermark_latents(
            device=self.device,
            radius=self.ring_radius,
            radius_cutoff=self.ring_radius_cutoff,
            watermark_region_mask=self.ring_masks,
            watermark_channel=self.ring_channels,
            original_latents=output_latents,
            watermark_pattern=self.ring_candidate_patterns[self.ring_current_user_id],
        )

        self.output_latents = output_latents
        if self.debug:
            self.visualize_hybrid_latents(output_latents, "after_injection.png")
        return output_latents

    def _diffusion_inverse(self, watermark_sd):
        hw_stride = 64 // self.gs_hw
        h_split = torch.cat(torch.split(watermark_sd, hw_stride, dim=1), dim=0)
        w_split = torch.cat(torch.split(h_split, hw_stride, dim=2), dim=0)
        vote = torch.sum(w_split, dim=0)
        res = torch.zeros_like(vote)
        res[vote > self.gs_threshold_vote] = 1
        return res

    def _eval_gs(self, latents):
        extracted_ch = latents[0, self.gs_channels]
        reversed_m = (extracted_ch > 0).int()
        if self.use_chacha:
            reversed_bits = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
            reversed_sd = (
                torch.from_numpy(reversed_bits)
                .to(self.device)
                .reshape(len(self.gs_channels), 64, 64)
            )
        else:
            reversed_sd = (reversed_m + self.gs_key) % 2
        channel_accs = []
        for i in range(len(self.gs_channels)):
            reversed_watermark = self._diffusion_inverse(reversed_sd[i : i + 1])
            acc = (reversed_watermark == self.gs_msg[i]).float().mean().item()
            channel_accs.append(acc)

        avg_acc = np.mean(channel_accs)
        is_detected = avg_acc >= self.tau_gs
        is_traceable = avg_acc >= self.tau_gs_traceable
        return is_detected, is_traceable, avg_acc

    def _eval_ring(self, latents):
        latents_fft = fft(latents)
        distances = []
        for pattern in self.ring_candidate_patterns:
            d = get_distance(
                pattern,
                latents_fft,
                self.ring_masks,
                channel=self.ring_channels,
                p=1,
                mode="complex",
            )
            distances.append(d / len(self.ring_channels))
        best_match_idx = int(np.argmin(distances))
        min_dist = distances[best_match_idx]
        is_match = best_match_idx == self.ring_current_user_id
        if self.debug:
            print(self.ring_current_user_id, best_match_idx, min_dist, min(distances))
        is_detected = min_dist < self.ring_threshold_dist
        is_traceable = is_detected and is_match
        return is_detected, is_traceable, min_dist, is_match

    @torch.no_grad()
    def eval_watermark(self, reversed_latents):
        is_gs_detected, is_gs_traceable, gs_acc = self._eval_gs(reversed_latents)
        is_ring_detected, is_ring_traceable, ring_dist, is_ring_match = self._eval_ring(
            reversed_latents
        )

        if self.debug:
            print("Gaussian Shading Acc: ", gs_acc)
            print("RingID L1 Distance: ", ring_dist)
            print("RingID Match: ", is_ring_match)
            print("GS & RID Detected:", is_gs_detected, is_ring_detected)
            print("GS & RID Traceable:", is_gs_traceable, is_ring_traceable)

        if is_gs_detected or is_ring_detected:
            self.tp_detection_count += 1

        if (is_gs_detected or is_ring_detected) and (
            is_gs_traceable or is_ring_traceable
        ):
            self.tp_traceability_count += 1

        return gs_acc, float(is_ring_match)

    def get_tpr(self):
        return self.tp_detection_count, self.tp_traceability_count
