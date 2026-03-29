"""
╔══════════════════════════════════════════════════════════════════════════╗
║        CubeSat ACS — SDR GFSK Decoding Algorithm                        ║
║                                                                          ║
║  Author  : Arya MGC                                                      ║
║  Version : 2.1.0                                                         ║
║                                                                          ║
║  Full receive chain for GFSK-modulated CubeSat telemetry downlink:       ║
║                                                                          ║
║   IQ Samples                                                             ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [DC Offset & IQ Imbalance Correction]                                  ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Carrier Frequency Offset (CFO) Estimation & Correction]               ║
║      │         └─ FFT-based coarse + Costas loop fine                    ║
║      ▼                                                                   ║
║   [Matched Filter  — Root Raised Cosine / Gaussian]                      ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Symbol Timing Recovery — Mueller & Müller TED]                        ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [FM Discriminator  — Differential Phase Demodulation]                  ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Bit Slicer  → NRZ bits]                                               ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Preamble & SFD Detection  — Correlator]                               ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Frame Synchronisation & Packet Extraction]                            ║
║      │                                                                   ║
║      ▼                                                                   ║
║   [Checksum Verification  → TelemetryPacket]                             ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import struct
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, lfilter, correlate
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import logging
import time

log = logging.getLogger("SDR-Decoder")


# ─── Decoder Configuration ────────────────────────────────────────────────────

@dataclass
class DecoderConfig:
    """All tunable parameters for the SDR receive chain."""
    # RF / baseband
    sample_rate:    int   = 2_000_000      # Hz  (RTL-SDR typical)
    center_freq:    float = 433.0e6        # Hz
    bit_rate:       int   = 250_000        # bps  (NRF24L01+ 250 kbps)
    freq_dev:       float = 125_000.0      # Hz  (modulation index h = 0.5)
    bt_product:     float = 0.5            # Bandwidth-Time product (GFSK)

    # Derived (computed post-init)
    sps:            int   = 0              # samples per symbol
    h:              float = 0.0            # modulation index

    # Frame format
    preamble_bits:  int   = 8             # 0xAA preamble length in bits
    sfd_pattern:    int   = 0xAE          # Start-of-Frame Delimiter (header byte)
    payload_bytes:  int   = 32            # NRF24L01+ max payload

    # Timing recovery (Mueller & Müller)
    mm_gain:        float = 0.01          # TED loop gain
    mm_alpha:       float = 0.01          # proportional
    mm_beta:        float = 0.001         # integral

    # CFO correction (Costas loop)
    costas_alpha:   float = 0.01
    costas_beta:    float = 0.001

    # SNR / quality thresholds
    min_snr_db:     float = 5.0
    sync_threshold: float = 0.6           # Normalised correlation threshold

    def __post_init__(self):
        self.sps = self.sample_rate // self.bit_rate
        self.h   = 2 * self.freq_dev / self.bit_rate


# ─── Stage 1 — IQ Pre-processing ─────────────────────────────────────────────

class IQPreprocessor:
    """
    Correct hardware imperfections before any demodulation.

    Corrections applied:
      1. DC offset removal  (mean subtraction per block)
      2. IQ imbalance compensation  (amplitude + phase mismatch)
      3. Anti-aliasing low-pass filter
    """

    def __init__(self, cfg: DecoderConfig):
        self.cfg    = cfg
        self._dc_i  = 0.0
        self._dc_q  = 0.0
        self._alpha = 0.001     # DC tracking smoothing

        # Anti-alias LPF: pass up to bit_rate * 2, stop at sample_rate/2
        cutoff = cfg.bit_rate * 3.0 / (cfg.sample_rate / 2)
        cutoff = min(cutoff, 0.99)
        self._b, self._a = butter(5, cutoff, btype='low')

    def remove_dc(self, iq: np.ndarray) -> np.ndarray:
        """Adaptive DC offset removal using leaky integrator."""
        dc = np.mean(iq)
        self._dc_i = (1 - self._alpha) * self._dc_i + self._alpha * dc.real
        self._dc_q = (1 - self._alpha) * self._dc_q + self._alpha * dc.imag
        return iq - complex(self._dc_i, self._dc_q)

    def iq_imbalance_correct(self, iq: np.ndarray,
                              amplitude_err: float = 0.0,
                              phase_err_deg: float = 0.0) -> np.ndarray:
        """
        Correct IQ amplitude and phase imbalance.
        amplitude_err: fractional gain error of Q channel (e.g. 0.02 = 2%)
        phase_err_deg: phase quadrature error in degrees
        """
        phi   = np.deg2rad(phase_err_deg)
        alpha = 1.0 + amplitude_err

        # Compensation matrix
        i_out = iq.real - iq.imag * np.tan(phi)
        q_out = iq.imag / (alpha * np.cos(phi))
        return i_out + 1j * q_out

    def lowpass_filter(self, iq: np.ndarray) -> np.ndarray:
        """5th-order Butterworth anti-alias LPF."""
        i_filt = filtfilt(self._b, self._a, iq.real)
        q_filt = filtfilt(self._b, self._a, iq.imag)
        return i_filt + 1j * q_filt

    def process(self, iq: np.ndarray,
                amplitude_err: float = 0.0,
                phase_err_deg: float = 0.0) -> np.ndarray:
        """Full pre-processing pipeline."""
        iq = self.remove_dc(iq)
        iq = self.iq_imbalance_correct(iq, amplitude_err, phase_err_deg)
        iq = self.lowpass_filter(iq)
        return iq

    def estimate_snr(self, iq: np.ndarray) -> float:
        """
        Estimate SNR using the signal variance method.
        SNR_dB = 10·log10(signal_power / noise_floor)
        """
        mag    = np.abs(iq)
        sig_pwr = np.mean(mag)**2
        noi_pwr = np.var(mag)
        if noi_pwr < 1e-20:
            return 60.0
        return 10 * np.log10(sig_pwr / noi_pwr)


# ─── Stage 2 — Carrier Frequency Offset Correction ───────────────────────────

class CFOCorrector:
    """
    Two-stage Carrier Frequency Offset (CFO) estimation & correction.

    Stage A — Coarse: FFT-based frequency estimation on the received signal
               squared (removes BPSK/FSK modulation, exposing carrier).
    Stage B — Fine:   Costas loop for residual CFO tracking.

    For GFSK the squaring trick doesn't remove modulation cleanly, so we use
    the fourth-power method which works on any constant-envelope signal.
    """

    def __init__(self, cfg: DecoderConfig):
        self.cfg         = cfg
        self._phase      = 0.0         # Costas loop phase accumulator
        self._freq_err   = 0.0         # Costas loop frequency error
        self._alpha      = cfg.costas_alpha
        self._beta       = cfg.costas_beta

    # ── Coarse CFO (FFT method) ────────────────────────────────────────────
    def estimate_cfo_fft(self, iq: np.ndarray) -> float:
        """
        Estimate CFO by looking at the spectral peak of iq^4.
        Returns CFO estimate in Hz.
        """
        iq4    = iq ** 4                          # Remove GFSK modulation
        N      = len(iq4)
        fft    = np.fft.fftshift(np.fft.fft(iq4, N))
        freqs  = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/self.cfg.sample_rate))
        peak   = freqs[np.argmax(np.abs(fft))]
        return peak / 4.0                          # Undo the x4 scaling

    def correct_coarse(self, iq: np.ndarray, cfo_hz: float) -> np.ndarray:
        """Apply frequency shift to compensate estimated CFO."""
        t      = np.arange(len(iq)) / self.cfg.sample_rate
        shift  = np.exp(-1j * 2 * np.pi * cfo_hz * t)
        return iq * shift

    # ── Fine CFO tracking (Costas loop) ───────────────────────────────────
    def costas_loop(self, iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Second-order Costas PLL for residual CFO and carrier phase tracking.

        Returns:
            corrected IQ samples
            phase error trajectory (for diagnostics)
        """
        out       = np.zeros_like(iq)
        phase_log = np.zeros(len(iq))
        phase     = self._phase
        freq_err  = self._freq_err

        for n, s in enumerate(iq):
            # Apply current phase correction
            corrected     = s * np.exp(-1j * phase)
            out[n]        = corrected

            # Phase detector: sign(I) * Q  (decision-directed)
            phase_err     = np.sign(corrected.real) * corrected.imag

            # Loop filter (proportional + integral)
            freq_err     += self._beta * phase_err
            phase        += self._alpha * phase_err + freq_err
            phase_log[n]  = phase

        self._phase    = phase
        self._freq_err = freq_err
        return out, phase_log

    def process(self, iq: np.ndarray) -> Tuple[np.ndarray, float]:
        """Full CFO correction: coarse FFT then fine Costas tracking."""
        cfo_hz = self.estimate_cfo_fft(iq)
        log.debug("Coarse CFO estimate: %.1f Hz", cfo_hz)
        iq_coarse = self.correct_coarse(iq, cfo_hz)
        iq_fine, _ = self.costas_loop(iq_coarse)
        return iq_fine, cfo_hz


# ─── Stage 3 — FM Discriminator (GFSK Demodulation) ─────────────────────────

class FMDiscriminator:
    """
    Differential phase FM discriminator for GFSK demodulation.

    Instantaneous frequency:
        f(n) = angle(x(n) · conj(x(n-1))) · Fs / (2π)

    Normalised to ±1 relative to the frequency deviation.
    Also implements an optional limiter (hard AGC) before discrimination.
    """

    def __init__(self, cfg: DecoderConfig):
        self.cfg  = cfg
        self._prev = complex(1.0, 0.0)

        # Low-pass filter on discriminator output to reduce ISI
        cutoff = (cfg.bit_rate * 1.2) / (cfg.sample_rate / 2)
        cutoff = min(cutoff, 0.99)
        self._b, self._a = butter(3, cutoff)

    def discriminate(self, iq: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous frequency from IQ samples.
        Returns normalised frequency deviation in [-1, +1].
        """
        # Hard limiter: normalise amplitude (removes AM noise)
        magnitude = np.abs(iq)
        magnitude = np.where(magnitude < 1e-12, 1e-12, magnitude)
        iq_norm   = iq / magnitude

        # Differential phase
        delayed   = np.concatenate([[self._prev], iq_norm[:-1]])
        self._prev = iq_norm[-1]
        diff_phase = iq_norm * np.conj(delayed)

        # Instantaneous frequency [Hz]
        inst_freq  = np.angle(diff_phase) * self.cfg.sample_rate / (2 * np.pi)

        # Normalise to ±1 relative to frequency deviation
        normalised = inst_freq / self.cfg.freq_dev
        normalised = np.clip(normalised, -2.0, 2.0)   # clip hard outliers

        # Post-discriminator LPF
        filtered   = lfilter(self._b, self._a, normalised)
        return filtered

    def compute_eye_diagram(self, demod: np.ndarray,
                             sps: int) -> np.ndarray:
        """
        Build eye diagram matrix (num_eyes × 2*sps) for BER estimation.
        Each row is one symbol period overlaid.
        """
        trim = len(demod) - len(demod) % (2 * sps)
        mat  = demod[:trim].reshape(-1, 2 * sps)
        return mat


# ─── Stage 4 — Symbol Timing Recovery (Mueller & Müller) ─────────────────────

class TimingRecovery:
    """
    Mueller & Müller symbol timing error detector (TED).

    The M&M algorithm is a decision-directed TED that works on the
    demodulated (post-discriminator) real signal. It adjusts the sampling
    instant to maximise eye opening.

    Loop architecture:
        e(n) = d̂(n-1)·x(n) - d̂(n)·x(n-1)      <- timing error
        τ(n) = τ(n-1) + α·e(n) + β·∑e(n)         <- timing correction

    where d̂(n) is the hard decision on the current sample.
    """

    def __init__(self, cfg: DecoderConfig):
        self.cfg         = cfg
        self._mu         = 0.0           # fractional timing offset ∈ [0, sps)
        self._mu_int     = 0             # integer part
        self._int_err    = 0.0           # integral of timing error
        self._prev_x     = 0.0           # x(n-1)
        self._prev_d     = 0.0           # d̂(n-1)
        self._alpha      = cfg.mm_alpha
        self._beta       = cfg.mm_beta
        self.sps         = cfg.sps

    def _interpolate(self, buf: np.ndarray, mu: float) -> float:
        """
        Linear interpolation between samples.
        For production use cubic (Farrow) interpolation — implemented below.
        """
        n = int(mu)
        f = mu - n
        if n + 1 < len(buf):
            return (1 - f) * buf[n] + f * buf[n + 1]
        return buf[n] if n < len(buf) else 0.0

    def _cubic_interpolate(self, buf: np.ndarray, mu: float) -> float:
        """
        Cubic Farrow interpolator (4-tap).
        More accurate than linear for low oversampling ratios.
        """
        n = int(mu)
        f = mu - n
        # Gather 4 surrounding samples
        samples = [buf[max(n - 1, 0)],
                   buf[n] if n < len(buf) else 0.0,
                   buf[n + 1] if n + 1 < len(buf) else 0.0,
                   buf[n + 2] if n + 2 < len(buf) else 0.0]
        # Cubic Lagrange
        y  = (-f*(1-f)*(2-f)/6)         * samples[0]
        y += ((1-f*f)*(2-f)/2)          * samples[1]
        y += (f*(1+f)*(2-f)/2)          * samples[2]
        y += (-f*(1-f*f)/6)             * samples[3]
        return y

    def recover(self, demod: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run M&M timing recovery on the discriminator output.

        Returns:
            symbols  : downsampled symbol stream (one sample per bit)
            ted_err  : timing error trajectory (diagnostics)
        """
        symbols   = []
        ted_errors = []
        buf        = demod
        k          = self.sps                 # current sample pointer

        while k + self.sps < len(buf):
            # Interpolate at current timing estimate
            x = self._cubic_interpolate(buf, k + self._mu)

            # Hard decision
            d = 1.0 if x > 0 else -1.0
            symbols.append(x)

            # M&M timing error
            e = self._prev_d * x - d * self._prev_x

            # Loop filter
            self._int_err += self._beta * e
            delta_mu       = self._alpha * e + self._int_err

            # Update timing
            self._mu      += delta_mu
            self._mu       = np.clip(self._mu, -self.sps/2, self.sps/2)

            self._prev_x = x
            self._prev_d = d
            ted_errors.append(e)

            k += self.sps

        return np.array(symbols), np.array(ted_errors)


# ─── Stage 5 — Bit Slicing & NRZ Decoding ────────────────────────────────────

class BitSlicer:
    """
    Convert soft symbol stream → hard NRZ bits.

    Also applies optional NRZI decoding (used by some NRF configurations)
    and bit-stuffing removal.
    """

    def __init__(self, threshold: float = 0.0, nrzi: bool = False):
        self.threshold = threshold
        self.nrzi      = nrzi
        self._prev_bit = 0

    def slice(self, symbols: np.ndarray) -> np.ndarray:
        """Hard decision: symbol > threshold → 1, else → 0."""
        bits = (symbols > self.threshold).astype(np.uint8)
        if self.nrzi:
            bits = self._decode_nrzi(bits)
        return bits

    def _decode_nrzi(self, bits: np.ndarray) -> np.ndarray:
        """
        NRZI decode: bit = 1 if transition from previous, else 0.
        Some NRF24L01+ variants transmit NRZI-encoded data.
        """
        decoded     = np.zeros_like(bits)
        prev        = self._prev_bit
        for i, b in enumerate(bits):
            decoded[i] = 1 if b != prev else 0
            prev = b
        self._prev_bit = prev
        return decoded

    def bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Pack bit array into bytes (MSB first)."""
        pad   = (8 - len(bits) % 8) % 8
        bits  = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    def compute_ber(self, rx_bits: np.ndarray,
                    tx_bits: np.ndarray) -> float:
        """Compute bit error rate against known transmitted sequence."""
        length = min(len(rx_bits), len(tx_bits))
        errors = np.sum(rx_bits[:length] != tx_bits[:length])
        return errors / length if length > 0 else 1.0


# ─── Stage 6 — Preamble & Frame Synchronisation ──────────────────────────────

class FrameSync:
    """
    Locate packet boundaries within the bit stream.

    NRF24L01+ frame structure:
        [Preamble: 0xAA / 0x55 × N bits]
        [Address: 3–5 bytes]
        [Payload: up to 32 bytes]
        [CRC: 1–2 bytes]

    We search for the SFD byte (0xAE) by:
      1. Sliding-window cross-correlation with preamble template.
      2. Threshold the correlation to find candidate sync positions.
      3. Try parsing a packet at each candidate → validate checksum.
    """

    PREAMBLE_PATTERNS = {
        'AA': np.array([1,0,1,0,1,0,1,0], dtype=np.float32),
        '55': np.array([0,1,0,1,0,1,0,1], dtype=np.float32),
    }

    def __init__(self, cfg: DecoderConfig):
        self.cfg       = cfg
        self.sfd_bits  = np.unpackbits(
            np.array([cfg.sfd_pattern], dtype=np.uint8)
        ).astype(np.float32) * 2 - 1    # NRZ: 0→-1, 1→+1

        # Build correlation template: preamble + SFD
        preamble = self.PREAMBLE_PATTERNS['AA'] * 2 - 1   # NRZ
        self._template = np.concatenate([preamble, self.sfd_bits])

    def find_sync_positions(self, bits: np.ndarray,
                             threshold: Optional[float] = None) -> List[int]:
        """
        Cross-correlate bit stream with preamble+SFD template.
        Returns list of candidate packet start positions (bit indices).
        """
        thr     = threshold or self.cfg.sync_threshold
        nrz     = bits.astype(np.float32) * 2 - 1         # NRZ convert
        tpl     = self._template
        L       = len(tpl)

        # Normalised cross-correlation
        corr    = correlate(nrz, tpl, mode='full')
        norm    = np.max(np.abs(corr)) + 1e-12
        corr_n  = corr / norm

        # Find peaks above threshold
        lags     = np.arange(-(L-1), len(nrz))
        peaks    = np.where(corr_n > thr)[0]
        positions = []
        last      = -999

        for p in peaks:
            start = lags[p] + L       # byte-aligned offset after SFD
            if start > last + 16 and 0 <= start < len(bits):
                positions.append(start)
                last = start

        log.debug("FrameSync: found %d candidate positions", len(positions))
        return positions

    def extract_packet(self, bits: np.ndarray,
                        start: int) -> Optional[bytes]:
        """
        Extract payload_bytes starting at bit position `start`.
        Returns raw bytes if enough bits remain, else None.
        """
        end  = start + self.cfg.payload_bytes * 8
        if end > len(bits):
            return None
        chunk = bits[start:end].astype(np.uint8)
        return np.packbits(chunk).tobytes()


# ─── Stage 7 — Packet Validator ──────────────────────────────────────────────

@dataclass
class DecodedPacket:
    """Fully decoded and verified telemetry packet."""
    seq:        int
    altitude:   float    # m
    velocity:   float    # m/s
    roll:       float    # rad
    pitch:      float    # rad
    yaw:        float    # rad
    timestamp:  int      # ms
    rssi:       float    # dBm (embedded measurement)
    rx_rssi:    float    # dBm (estimated from signal level)
    snr_db:     float    # dB
    cfo_hz:     float    # Hz  (CFO at acquisition)
    valid:      bool     = True
    error_msg:  str      = ""

    def as_dict(self) -> Dict:
        return {
            'seq':       self.seq,
            'altitude_km': self.altitude / 1e3,
            'velocity_ms': self.velocity,
            'roll_deg':  np.degrees(self.roll),
            'pitch_deg': np.degrees(self.pitch),
            'yaw_deg':   np.degrees(self.yaw),
            'timestamp_ms': self.timestamp,
            'rssi_dbm':  self.rssi,
            'snr_db':    self.snr_db,
            'cfo_hz':    self.cfo_hz,
        }


class PacketValidator:
    """
    Validates extracted bytes against the NRF24L01+ telemetry frame format.

    Packet (32 bytes):
      [0]   0xAE  header
      [1]   seq   uint8
      [2:6] alt   float32
      [6:10] vel  float32
      [10:14] roll float32
      [14:18] pitch float32
      [18:22] yaw  float32
      [22:26] ts   uint32
      [26:30] rssi float32
      [30]  chk   uint8  (sum of bytes 1..29 mod 256)
      [31]  0xEF  footer
    """

    HEADER  = 0xAE
    FOOTER  = 0xEF
    MIN_ALT = 200_000.0      # m  (sanity: LEO minimum ~200 km)
    MAX_ALT = 2_000_000.0    # m  (LEO maximum ~2000 km)
    MAX_VEL = 1_000.0        # m/s (generous bound for delta-v manoeuvres)
    MAX_ATT = np.pi          # rad

    def validate(self, raw: bytes,
                 rx_rssi: float = -70.0,
                 snr_db: float  = 15.0,
                 cfo_hz: float  = 0.0) -> Optional[DecodedPacket]:
        """
        Full packet validation pipeline.
        Returns DecodedPacket on success, None on failure.
        """
        # 1. Length check
        if len(raw) < 32:
            log.debug("Packet too short: %d bytes", len(raw))
            return None

        raw = raw[:32]

        # 2. Header / footer
        if raw[0] != self.HEADER:
            log.debug("Bad header: 0x%02X", raw[0])
            return None
        if raw[31] != self.FOOTER:
            log.debug("Bad footer: 0x%02X", raw[31])
            return None

        # 3. Checksum (sum of payload bytes 1..29 mod 256)
        chk_calc = sum(raw[1:30]) & 0xFF
        chk_rx   = raw[30]
        if chk_calc != chk_rx:
            log.debug("Checksum FAIL: calc=0x%02X rx=0x%02X", chk_calc, chk_rx)
            return None

        # 4. Unpack fields
        try:
            (seq, alt, vel,
             roll, pitch, yaw,
             ts, rssi) = struct.unpack_from('<BffffffIf', raw, 1)
        except struct.error as e:
            log.debug("Unpack error: %s", e)
            return None

        # 5. Sanity bounds
        if not (self.MIN_ALT <= alt <= self.MAX_ALT):
            log.debug("Altitude out of bounds: %.1f m", alt)
            return None
        if abs(vel) > self.MAX_VEL:
            log.debug("Velocity out of bounds: %.2f m/s", vel)
            return None
        for angle, name in [(roll,'roll'),(pitch,'pitch'),(yaw,'yaw')]:
            if abs(angle) > self.MAX_ATT:
                log.debug("Attitude out of bounds (%s): %.3f rad", name, angle)
                return None

        return DecodedPacket(
            seq       = seq,
            altitude  = alt,
            velocity  = vel,
            roll      = roll,
            pitch     = pitch,
            yaw       = yaw,
            timestamp = ts,
            rssi      = rssi,
            rx_rssi   = rx_rssi,
            snr_db    = snr_db,
            cfo_hz    = cfo_hz,
        )


# ─── Full Decoder Pipeline ────────────────────────────────────────────────────

class SDRDecoder:
    """
    Orchestrates all stages of the SDR receive chain.

    Usage:
        cfg     = DecoderConfig()
        decoder = SDRDecoder(cfg)
        packets = decoder.decode(iq_samples)

    The decoder is stateful — call decode() repeatedly with successive
    blocks of IQ samples (e.g. from RTL-SDR or file).
    """

    def __init__(self, cfg: Optional[DecoderConfig] = None):
        self.cfg        = cfg or DecoderConfig()
        self.pre        = IQPreprocessor(self.cfg)
        self.cfo        = CFOCorrector(self.cfg)
        self.discrim    = FMDiscriminator(self.cfg)
        self.timing     = TimingRecovery(self.cfg)
        self.slicer     = BitSlicer(threshold=0.0, nrzi=False)
        self.framesync  = FrameSync(self.cfg)
        self.validator  = PacketValidator()

        # Diagnostics
        self.stats: Dict = {
            'blocks_processed': 0,
            'packets_decoded':  0,
            'packets_failed':   0,
            'avg_snr_db':       0.0,
            'avg_cfo_hz':       0.0,
        }
        self._snr_acc   = 0.0
        self._cfo_acc   = 0.0

    def decode(self, iq: np.ndarray) -> List[DecodedPacket]:
        """
        Main entry point. Process one block of IQ samples.

        Args:
            iq: complex64 array of IQ samples at self.cfg.sample_rate

        Returns:
            List of successfully decoded and validated DecodedPacket objects.
        """
        self.stats['blocks_processed'] += 1
        results = []

        # ── Stage 1: IQ pre-processing ─────────────────────────────────
        iq = self.pre.process(iq)
        snr = self.pre.estimate_snr(iq)

        if snr < self.cfg.min_snr_db:
            log.debug("SNR too low: %.1f dB — skipping block", snr)
            return results

        # ── Stage 2: CFO correction ────────────────────────────────────
        iq, cfo_hz = self.cfo.process(iq)

        # ── Stage 3: FM discriminator ──────────────────────────────────
        demod = self.discrim.discriminate(iq)

        # ── Stage 4: Symbol timing recovery ───────────────────────────
        symbols, _ted = self.timing.recover(demod)

        # ── Stage 5: Bit slicing ───────────────────────────────────────
        bits = self.slicer.slice(symbols)

        # ── Stage 6: Frame synchronisation ────────────────────────────
        sync_positions = self.framesync.find_sync_positions(bits)

        # ── Stage 7: Extract & validate packets ───────────────────────
        rx_rssi = -65.0 + snr           # estimate from signal level
        for pos in sync_positions:
            raw = self.framesync.extract_packet(bits, pos)
            if raw is None:
                continue
            pkt = self.validator.validate(raw,
                                           rx_rssi=rx_rssi,
                                           snr_db=snr,
                                           cfo_hz=cfo_hz)
            if pkt:
                results.append(pkt)
                self.stats['packets_decoded'] += 1
                log.info("PKT seq=%-4d  alt=%8.1fkm  vel=%+7.3fm/s  "
                         "SNR=%5.1fdB  CFO=%+8.1fHz",
                         pkt.seq, pkt.altitude/1e3,
                         pkt.velocity, snr, cfo_hz)
            else:
                self.stats['packets_failed'] += 1

        # ── Update diagnostics ─────────────────────────────────────────
        n = self.stats['blocks_processed']
        self._snr_acc += snr
        self._cfo_acc += abs(cfo_hz)
        self.stats['avg_snr_db']  = self._snr_acc / n
        self.stats['avg_cfo_hz']  = self._cfo_acc / n

        return results

    def print_stats(self):
        print("\n" + "═"*50)
        print("  SDR Decoder — Session Statistics")
        print("═"*50)
        s = self.stats
        total = s['packets_decoded'] + s['packets_failed']
        rate  = s['packets_decoded'] / total * 100 if total > 0 else 0
        print(f"  Blocks processed  : {s['blocks_processed']}")
        print(f"  Packets decoded   : {s['packets_decoded']}")
        print(f"  Packets failed    : {s['packets_failed']}")
        print(f"  Decode success    : {rate:.1f} %")
        print(f"  Average SNR       : {s['avg_snr_db']:.1f} dB")
        print(f"  Average CFO       : {s['avg_cfo_hz']:.1f} Hz")
        print("═"*50 + "\n")


# ─── Loopback Test / Demo ─────────────────────────────────────────────────────

def run_loopback_test(num_packets: int = 20, snr_db: float = 18.0) -> None:
    """
    End-to-end loopback test: modulate → add noise → decode.
    Validates the full chain without real hardware.
    """
    import sys
    sys.path.insert(0, '.')

    # Import modulator from telemetry module
    try:
        from telemetry_sdr_nrf import (GFSKModulator, TelemetryPacket)
        modulator = GFSKModulator()
    except ImportError:
        # Inline minimal modulator for standalone test
        class _Mod:
            def __init__(self):
                self.cfg = DecoderConfig()
            def modulate_packet(self, pkt_bytes):
                bits   = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
                nrz    = bits.astype(np.float64) * 2 - 1
                sps    = self.cfg.sps
                up     = np.repeat(nrz, sps)
                phase  = 2*np.pi*self.cfg.freq_dev/self.cfg.sample_rate
                return np.exp(1j * np.cumsum(up) * phase)
        modulator = _Mod()

    cfg     = DecoderConfig()
    decoder = SDRDecoder(cfg)

    print("═"*55)
    print("  SDR Decoder — End-to-End Loopback Test")
    print(f"  {num_packets} packets  |  SNR = {snr_db:.0f} dB")
    print("═"*55)

    decoded_count = 0

    for i in range(num_packets):
        # Build a fake telemetry packet
        alt   = 400_000.0 + np.random.uniform(-5000, 5000)
        vel   = np.random.uniform(-2.0, 2.0)
        roll  = np.random.uniform(-0.05, 0.05)
        pitch = np.random.uniform(-0.05, 0.05)
        yaw   = np.random.uniform(-0.02, 0.02)

        payload = struct.pack('<BffffffIf',
                              i & 0xFF,
                              alt, vel, roll, pitch, yaw,
                              i * 20, -65.0)
        chk  = sum(payload) & 0xFF
        raw  = bytes([0xAE]) + payload + bytes([chk, 0xEF])

        # Modulate
        bits   = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        nrz    = bits.astype(np.float64) * 2 - 1
        up     = np.repeat(nrz, cfg.sps)
        phase  = 2 * np.pi * cfg.freq_dev / cfg.sample_rate
        iq     = np.exp(1j * np.cumsum(up) * phase)

        # Add preamble context (helps correlator)
        pream  = np.tile([1, -1], cfg.sps * 4)
        iq     = np.concatenate([np.exp(1j * np.cumsum(pream) * phase), iq])

        # AWGN channel
        snr_lin = 10 ** (snr_db / 10)
        pwr     = np.mean(np.abs(iq)**2)
        n0      = pwr / snr_lin
        noise   = np.sqrt(n0/2) * (np.random.randn(len(iq)) +
                                    1j * np.random.randn(len(iq)))
        rx_iq   = iq + noise

        # Add small CFO (±500 Hz)
        cfo_true = np.random.uniform(-500, 500)
        t        = np.arange(len(rx_iq)) / cfg.sample_rate
        rx_iq   *= np.exp(1j * 2 * np.pi * cfo_true * t)

        # Decode
        pkts = decoder.decode(rx_iq)
        if pkts:
            p = pkts[0]
            err_alt = abs(p.altitude - alt)
            print(f"  [PKT {i:3d}] seq={p.seq:3d}  "
                  f"alt_err={err_alt:.0f}m  "
                  f"SNR={p.snr_db:.1f}dB  "
                  f"CFO_est={p.cfo_hz:+.0f}Hz  "
                  f"CFO_true={cfo_true:+.0f}Hz  ✓")
            decoded_count += 1
        else:
            print(f"  [PKT {i:3d}] ✗ — not decoded")

    decoder.print_stats()
    print(f"  Loopback decode rate: {decoded_count}/{num_packets} "
          f"= {100*decoded_count/num_packets:.0f}%\n")


# ─── RTL-SDR File Reader (for offline processing) ─────────────────────────────

class RTLSDRFileReader:
    """
    Read IQ samples from RTL-SDR binary captures (.bin / .cs8 / .cf32).

    RTL-SDR binary format: interleaved uint8 I/Q, offset binary (127.5 = 0).
    Use rtl_sdr -f 433e6 -s 2e6 -n 4000000 capture.bin to record.
    """

    def __init__(self, filename: str, fmt: str = 'cs8'):
        """
        Args:
            filename: path to IQ file
            fmt: 'cs8'  = uint8 interleaved (RTL-SDR default)
                 'cf32' = float32 interleaved (GNU Radio / SDR#)
        """
        self.filename = filename
        self.fmt      = fmt

    def read(self, num_samples: int = 256_000,
             offset_samples: int = 0) -> np.ndarray:
        """Read a block of IQ samples as complex64."""
        if self.fmt == 'cs8':
            raw   = np.fromfile(self.filename, dtype=np.uint8,
                                count=num_samples * 2,
                                offset=offset_samples * 2)
            iq    = (raw[0::2].astype(np.float32) - 127.5 +
                     1j * (raw[1::2].astype(np.float32) - 127.5)) / 128.0
        elif self.fmt == 'cf32':
            raw   = np.fromfile(self.filename, dtype=np.float32,
                                count=num_samples * 2,
                                offset=offset_samples * 8)
            iq    = raw[0::2] + 1j * raw[1::2]
        else:
            raise ValueError(f"Unknown format: {self.fmt}")
        return iq

    def stream_decode(self, decoder: SDRDecoder,
                       block_size: int = 32_000) -> List[DecodedPacket]:
        """Stream-decode an entire file in blocks."""
        all_packets = []
        offset      = 0
        while True:
            iq = self.read(num_samples=block_size, offset_samples=offset)
            if len(iq) < block_size // 4:
                break
            pkts = decoder.decode(iq)
            all_packets.extend(pkts)
            offset += block_size
            log.info("Processed offset %d — total decoded: %d",
                     offset, len(all_packets))
        return all_packets


# ─── Diagnostic Plots ─────────────────────────────────────────────────────────

def plot_decode_pipeline(iq_raw: np.ndarray, cfg: DecoderConfig):
    """Visualise each stage of the decode chain on one figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    pre     = IQPreprocessor(cfg)
    cfo_blk = CFOCorrector(cfg)
    disc    = FMDiscriminator(cfg)
    timing  = TimingRecovery(cfg)
    slicer  = BitSlicer()

    iq1          = pre.process(iq_raw)
    iq2, cfo_est = cfo_blk.process(iq1)
    demod        = disc.discriminate(iq2)
    symbols, ted = timing.recover(demod)
    bits         = slicer.slice(symbols)
    eye          = disc.compute_eye_diagram(demod, cfg.sps)

    fig = plt.figure(figsize=(16, 12), facecolor='#0D1117')
    gs  = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.4)
    BG, PNL, TXT, ACC1, ACC2, ACC3 = ('#0D1117','#161B22','#C9D1D9',
                                       '#58A6FF','#FF7B72','#3FB950')

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(PNL)
        ax.tick_params(colors='#8B949E', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.grid(True, color='#21262D', alpha=0.5, linewidth=0.5)
        return ax

    # IQ constellation (before correction)
    ax = styled_ax(gs[0, 0])
    ax.scatter(iq_raw.real[:2000], iq_raw.imag[:2000],
               s=1, alpha=0.3, color=ACC2)
    ax.set_title('IQ — Raw', color=TXT, fontsize=9)
    ax.set_xlabel('I', color=TXT, fontsize=8); ax.set_ylabel('Q', color=TXT, fontsize=8)

    # IQ constellation (after correction)
    ax2 = styled_ax(gs[0, 1])
    ax2.scatter(iq2.real[:2000], iq2.imag[:2000],
                s=1, alpha=0.3, color=ACC1)
    ax2.set_title('IQ — After CFO Correction', color=TXT, fontsize=9)
    ax2.set_xlabel('I', color=TXT, fontsize=8)

    # PSD
    ax3 = styled_ax(gs[0, 2])
    f, psd = scipy_signal.welch(iq_raw, fs=cfg.sample_rate, nperseg=1024)
    ax3.semilogy(f/1e3, psd, color=ACC3, lw=1)
    ax3.set_title('PSD', color=TXT, fontsize=9)
    ax3.set_xlabel('Frequency [kHz]', color=TXT, fontsize=8)
    ax3.tick_params(colors='#8B949E')

    # Discriminator output
    ax4 = styled_ax(gs[1, 0])
    t_demod = np.arange(min(len(demod), 500)) / cfg.sample_rate * 1e6
    ax4.plot(t_demod, demod[:500], color=ACC1, lw=0.8)
    ax4.axhline(0, color='#30363D', lw=0.5)
    ax4.set_title('FM Discriminator Output', color=TXT, fontsize=9)
    ax4.set_xlabel('Time [µs]', color=TXT, fontsize=8)

    # Eye diagram
    ax5 = styled_ax(gs[1, 1])
    for row in eye[:80]:
        ax5.plot(row, color=ACC1, lw=0.3, alpha=0.3)
    ax5.set_title('Eye Diagram', color=TXT, fontsize=9)
    ax5.set_xlabel('Sample', color=TXT, fontsize=8)

    # M&M TED error
    ax6 = styled_ax(gs[1, 2])
    ax6.plot(ted[:300], color=ACC3, lw=0.8)
    ax6.axhline(0, color='#30363D', lw=0.5)
    ax6.set_title('M&M Timing Error', color=TXT, fontsize=9)
    ax6.set_xlabel('Symbol', color=TXT, fontsize=8)

    # Symbol histogram
    ax7 = styled_ax(gs[2, 0])
    ax7.hist(symbols, bins=50, color=ACC1, alpha=0.8, edgecolor='none')
    ax7.axvline(0, color=ACC2, lw=1.5, ls='--')
    ax7.set_title('Symbol Distribution', color=TXT, fontsize=9)
    ax7.set_xlabel('Amplitude', color=TXT, fontsize=8)

    # Decoded bits
    ax8 = styled_ax(gs[2, 1])
    ax8.step(np.arange(min(len(bits),128)), bits[:128],
             color=ACC3, lw=1.5, where='post')
    ax8.set_ylim(-0.2, 1.2); ax8.set_title('Decoded Bits (first 128)', color=TXT, fontsize=9)
    ax8.set_xlabel('Bit index', color=TXT, fontsize=8)

    # CFO over time (Costas phase)
    ax9 = styled_ax(gs[2, 2])
    _, phase_log = cfo_blk.costas_loop(iq1[:5000])
    ax9.plot(phase_log, color=ACC2, lw=0.8)
    ax9.set_title(f'Costas Loop Phase\nCFO ≈ {cfo_est:.0f} Hz', color=TXT, fontsize=9)
    ax9.set_xlabel('Sample', color=TXT, fontsize=8)

    fig.suptitle('SDR GFSK Decode Pipeline — Arya MGC',
                 color='#F0F6FC', fontsize=13, fontweight='bold')
    plt.savefig('sdr_decode_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor=BG)
    log.info("Saved: sdr_decode_pipeline.png")
    plt.show()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CubeSat SDR GFSK Decoder — Arya MGC')
    parser.add_argument('--loopback', action='store_true',
                        help='Run end-to-end loopback test')
    parser.add_argument('--file', type=str, default=None,
                        help='RTL-SDR .bin file to decode')
    parser.add_argument('--plot', action='store_true',
                        help='Show pipeline diagnostic plots')
    parser.add_argument('--snr', type=float, default=18.0,
                        help='Simulated SNR for loopback [dB]')
    parser.add_argument('--packets', type=int, default=20,
                        help='Number of packets for loopback test')
    args = parser.parse_args()

    if args.loopback or (not args.file):
        run_loopback_test(num_packets=args.packets, snr_db=args.snr)

    if args.file:
        cfg     = DecoderConfig()
        decoder = SDRDecoder(cfg)
        reader  = RTLSDRFileReader(args.file, fmt='cs8')
        packets = reader.stream_decode(decoder)
        decoder.print_stats()
        print(f"Total packets decoded from file: {len(packets)}")

    if args.plot:
        cfg = DecoderConfig()
        # Generate synthetic IQ for pipeline visualisation
        raw  = bytes([0xAE]) + struct.pack('<BffffffIf',
               42, 405000.0, 1.23, 0.01, -0.01, 0.005, 840, -65.0)
        raw += bytes([(sum(raw[1:]) & 0xFF), 0xEF])
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        nrz  = bits.astype(np.float64) * 2 - 1
        up   = np.repeat(nrz, cfg.sps)
        ph   = 2 * np.pi * cfg.freq_dev / cfg.sample_rate
        iq   = np.exp(1j * np.cumsum(up) * ph)
        # Add noise + CFO
        snr_lin = 10**(15/10); pwr=np.mean(np.abs(iq)**2)
        noise = np.sqrt(pwr/snr_lin/2)*(np.random.randn(len(iq))+1j*np.random.randn(len(iq)))
        t     = np.arange(len(iq))/cfg.sample_rate
        iq_rx = (iq + noise) * np.exp(1j*2*np.pi*300*t)
        plot_decode_pipeline(iq_rx, cfg)
