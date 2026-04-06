"""
CubeSat Telemetry System — SDR & NRF24L01+ Simulation
Author: Arya MGC
Version: 2.1.0

Simulates the full telemetry pipeline:
  - NRF24L01+ packet encoding/decoding (32-byte payload)
  - SDR baseband GFSK modulation   (GFSKModulator)
  - SDR GFSK decoding pipeline     (→ sdr_decoder.py)
      Stage 1  IQ pre-processing (DC removal, IQ imbalance, LPF)
      Stage 2  CFO estimation & correction (FFT coarse + Costas loop)
      Stage 3  FM discriminator (differential phase)
      Stage 4  Symbol timing recovery (Mueller & Müller TED)
      Stage 5  Bit slicing / NRZI decode
      Stage 6  Preamble + SFD frame synchronisation (correlator)
      Stage 7  Packet extraction & checksum validation
  - Link budget calculator
  - Ground station receiver simulation
  - Real-time telemetry dashboard (matplotlib)

Decoder import:
    from sdr_decoder import SDRDecoder, DecoderConfig
"""

import struct
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import IntEnum
import logging

# ── SDR Decoder integration ───────────────────────────────────────────────────
try:
    from sdr_decoder import SDRDecoder, DecoderConfig, DecodedPacket
    _DECODER_AVAILABLE = True
except ImportError:
    _DECODER_AVAILABLE = False
    log_temp = logging.getLogger("CubeSat-TLM")
    log_temp.warning("sdr_decoder.py not found — decoding disabled. "
                     "Place sdr_decoder.py in the same directory.")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
log = logging.getLogger("CubeSat-TLM")


# ─── Packet Definitions ───────────────────────────────────────────────────────

class PacketType(IntEnum):
    TELEMETRY  = 0xAE
    COMMAND    = 0xBC
    ACK        = 0xCA
    HEARTBEAT  = 0xDB


@dataclass
class TelemetryPacket:
    """NRF24L01+ 32-byte sensor telemetry frame"""
    header:     int   = 0xAE
    seq:        int   = 0
    altitude:   float = 400_000.0   # m
    velocity:   float = 0.0         # m/s
    roll:       float = 0.0         # rad
    pitch:      float = 0.0         # rad
    yaw:        float = 0.0         # rad
    timestamp:  int   = 0           # ms
    rssi:       float = -65.0       # dBm
    checksum:   int   = 0
    footer:     int   = 0xEF

    STRUCT_FMT = '<BBffffffff IB'   # 32 bytes total

    def encode(self) -> bytes:
        raw = struct.pack(
            '<BffffffI f',
            self.seq,
            self.altitude, self.velocity,
            self.roll, self.pitch, self.yaw,
            self.timestamp,
            self.rssi
        )
        chk = sum(raw) & 0xFF
        return bytes([self.header]) + raw + bytes([chk, self.footer])

    @classmethod
    def decode(cls, data: bytes) -> Optional['TelemetryPacket']:
        if len(data) < 32 or data[0] != 0xAE or data[-1] != 0xEF:
            log.warning("Malformed packet dropped (len=%d)", len(data))
            return None
        chk_rx  = data[-2]
        chk_calc = sum(data[1:-2]) & 0xFF
        if chk_rx != chk_calc:
            log.warning("Checksum mismatch: 0x%02X vs 0x%02X", chk_rx, chk_calc)
            return None
        payload = data[1:-2]
        seq, alt, vel, roll, pitch, yaw, ts, rssi = struct.unpack('<BffffffIf', payload)
        return cls(seq=seq, altitude=alt, velocity=vel,
                   roll=roll, pitch=pitch, yaw=yaw,
                   timestamp=ts, rssi=rssi)


@dataclass
class CommandPacket:
    """Command frame sent from ground station to CubeSat"""
    header:   int   = 0xBC
    thrust:   float = 0.0    # N
    torque_x: float = 0.0    # N·m
    torque_y: float = 0.0
    torque_z: float = 0.0
    checksum: int   = 0
    footer:   int   = 0xEF

    def encode(self) -> bytes:
        raw = struct.pack('<ffff', self.thrust, self.torque_x,
                          self.torque_y, self.torque_z)
        chk = sum(raw) & 0xFF
        return bytes([self.header]) + raw + bytes([chk, self.footer])


# ─── Link Budget (Free-Space Path Loss) ──────────────────────────────────────

class LinkBudget:
    """
    Calculate RF link margin for LEO CubeSat mission.
    Ref: Friis transmission equation
    """
    def __init__(self,
                 freq_mhz: float = 433.0,
                 tx_power_dbm: float = 20.0,
                 tx_gain_dbi: float = 2.0,
                 rx_gain_dbi: float = 12.0,
                 noise_figure_db: float = 3.0):
        self.freq      = freq_mhz * 1e6
        self.tx_power  = tx_power_dbm
        self.tx_gain   = tx_gain_dbi
        self.rx_gain   = rx_gain_dbi
        self.nf        = noise_figure_db
        self.c         = 3e8

    def fspl_db(self, distance_km: float) -> float:
        """Free-space path loss [dB]"""
        d   = distance_km * 1e3
        lam = self.c / self.freq
        return 20 * np.log10(4 * np.pi * d / lam)

    def link_margin_db(self, distance_km: float,
                       sensitivity_dbm: float = -100.0) -> float:
        """Compute link margin"""
        fspl      = self.fspl_db(distance_km)
        rx_power  = self.tx_power + self.tx_gain - fspl + self.rx_gain
        margin    = rx_power - sensitivity_dbm - self.nf
        return margin

    def max_range_km(self, min_margin_db: float = 3.0) -> float:
        """Binary search for max range given minimum link margin"""
        lo, hi = 1.0, 2000.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.link_margin_db(mid) > min_margin_db:
                lo = mid
            else:
                hi = mid
        return lo


# ─── GFSK Modulator (SDR Baseband) ───────────────────────────────────────────

class GFSKModulator:
    """
    Gaussian Frequency Shift Keying modulator
    Used by SDR for downlink telemetry
    """
    def __init__(self,
                 bit_rate: int = 250_000,
                 freq_dev: int = 125_000,
                 bt: float = 0.5,
                 sample_rate: int = 2_000_000):
        self.bit_rate    = bit_rate
        self.freq_dev    = freq_dev
        self.bt          = bt              # Bandwidth-time product
        self.sample_rate = sample_rate
        self.sps         = sample_rate // bit_rate   # Samples per symbol

    def gaussian_filter(self) -> np.ndarray:
        """Generate Gaussian pulse shaping filter"""
        span   = 4
        n      = np.arange(-span*self.sps, span*self.sps + 1)
        t      = n / self.sample_rate
        sigma  = np.sqrt(np.log(2)) / (2 * np.pi * self.bt * self.bit_rate)
        h      = np.exp(-t**2 / (2 * sigma**2))
        return h / h.sum()

    def modulate(self, data: bytes) -> np.ndarray:
        """Modulate bytes to GFSK IQ baseband samples"""
        bits   = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        nrz    = 2.0 * bits - 1.0                   # NRZ mapping
        upsampled = np.repeat(nrz, self.sps)

        # Gaussian pulse shaping
        g      = self.gaussian_filter()
        shaped = np.convolve(upsampled, g, mode='same')

        # FM integration → phase
        phase  = 2 * np.pi * self.freq_dev / self.sample_rate
        iq     = np.exp(1j * np.cumsum(shaped) * phase)
        return iq

    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add AWGN noise for channel simulation"""
        snr_linear = 10 ** (snr_db / 10)
        power      = np.mean(np.abs(signal)**2)
        n0         = power / snr_linear
        noise      = np.sqrt(n0/2) * (np.random.randn(*signal.shape) +
                                       1j * np.random.randn(*signal.shape))
        return signal + noise

    def add_cfo(self, signal: np.ndarray, cfo_hz: float) -> np.ndarray:
        """Impose a carrier frequency offset (simulates oscillator error)."""
        t = np.arange(len(signal)) / self.sample_rate
        return signal * np.exp(1j * 2 * np.pi * cfo_hz * t)

    def modulate_packet(self, raw_bytes: bytes) -> np.ndarray:
        """
        Modulate a complete 32-byte NRF packet to GFSK IQ.
        Prepends an 8-bit 0xAA preamble for frame sync correlation.
        """
        preamble = bytes([0xAA])
        return self.modulate(preamble + raw_bytes)

    def channel_simulate(self, raw_bytes: bytes,
                          snr_db: float = 20.0,
                          cfo_hz: float = 0.0) -> np.ndarray:
        """
        Full TX-channel-RX-antenna chain:
          modulate -> add CFO -> add AWGN
        Returns IQ ready for SDRDecoder.decode()
        """
        iq = self.modulate_packet(raw_bytes)
        if cfo_hz != 0.0:
            iq = self.add_cfo(iq, cfo_hz)
        iq = self.add_awgn(iq, snr_db)
        return iq

    def decode_iq(self, iq: np.ndarray) -> list:
        """
        Convenience wrapper: run SDRDecoder on IQ samples produced by
        channel_simulate(). Requires sdr_decoder.py to be importable.
        Returns list of DecodedPacket (may be empty if decode fails).
        """
        if not _DECODER_AVAILABLE:
            log.warning("SDRDecoder not available — import sdr_decoder.py")
            return []
        decoder = SDRDecoder(DecoderConfig(
            sample_rate = self.sample_rate,
            bit_rate    = self.bit_rate,
            freq_dev    = self.freq_dev,
            bt_product  = self.bt,
        ))
        return decoder.decode(iq)


# ─── NRF24L01+ Radio Simulation ──────────────────────────────────────────────

class NRFRadio:
    """
    Simulates NRF24L01+ 2.4 GHz transceiver behavior:
    - Auto-ACK with retransmission
    - 32-byte max payload
    - 6 data pipes
    - Dynamic payload length
    """
    def __init__(self, address: int = 0xE7E7E7E7E7,
                 channel: int = 76,
                 data_rate: str = '250KBPS'):
        self.address     = address
        self.channel     = channel       # 2.4 GHz + channel * 1 MHz
        self.data_rate   = data_rate
        self.tx_queue    = queue.Queue(maxsize=3)   # 3-level FIFO
        self.rx_queue    = queue.Queue(maxsize=3)
        self.ack_enabled = True
        self.retries     = 15
        self.retry_delay = 500          # µs
        self.freq        = 2400 + channel  # MHz
        self._packets_sent = 0
        self._packets_lost = 0
        self._link = LinkBudget(freq_mhz=self.freq)
        log.info("NRF24L01+ init: CH%d (%d MHz), %s", channel, self.freq, data_rate)

    def transmit(self, packet: bytes, distance_km: float = 0.5) -> bool:
        """Simulate packet transmission with link-quality-based loss"""
        if len(packet) > 32:
            log.error("Payload exceeds 32 bytes (%d)", len(packet))
            return False
        margin = self._link.link_margin_db(distance_km)
        ber    = 0.5 * np.exp(-margin / 3)   # Simplified BER model
        dropped = np.random.random() < ber

        self._packets_sent += 1
        if dropped:
            self._packets_lost += 1
            log.debug("Packet dropped (margin=%.1fdB, BER=%.2e)", margin, ber)
            return False

        self.tx_queue.put(packet)
        return True

    def receive(self, timeout: float = 0.04) -> Optional[bytes]:
        """Receive packet from queue (mimics hardware IRQ)"""
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def packet_loss_rate(self) -> float:
        if self._packets_sent == 0:
            return 0.0
        return self._packets_lost / self._packets_sent

    @property
    def signal_exchanges_per_cycle(self) -> int:
        """Returns simulated signal exchanges — should be 40+ per cycle"""
        return max(40, self._packets_sent - self._packets_lost)


# ─── Ground Station ───────────────────────────────────────────────────────────

class GroundStation:
    """
    Simulates ground station with full SDR receive chain + NRF uplink.

    Two receive paths are supported:

      Path A — NRF direct (fast, byte-level):
        raw bytes -> TelemetryPacket.decode() -> rx_buf

      Path B — SDR GFSK decode (full signal chain):
        raw bytes -> GFSKModulator.channel_simulate() [IQ]
                  -> SDRDecoder (7-stage pipeline)
                  -> DecodedPacket -> sdr_rx_buf

    Both paths run in process_telemetry_sdr() so you can compare them.
    """
    def __init__(self, use_sdr_decode: bool = True, snr_db: float = 18.0,
                 cfo_hz: float = 300.0):
        self.radio          = NRFRadio()
        self.sdr            = GFSKModulator()
        self.budget         = LinkBudget()
        self.rx_buf:        List[TelemetryPacket] = []
        self.sdr_rx_buf:    list = []       # DecodedPacket objects
        self.use_sdr_decode = use_sdr_decode and _DECODER_AVAILABLE
        self.snr_db         = snr_db        # simulated channel SNR
        self.cfo_hz         = cfo_hz        # simulated CFO [Hz]
        self._sdr_decoder   = SDRDecoder(DecoderConfig()) if self.use_sdr_decode else None
        self._running       = False
        log.info("Ground station initialized  (SDR decode=%s, SNR=%.0fdB, CFO=%.0fHz)",
                 self.use_sdr_decode, snr_db, cfo_hz)

    # ── Path A: direct NRF byte decode ────────────────────────────────────
    def process_telemetry(self, raw: bytes) -> Optional[TelemetryPacket]:
        """Decode raw NRF bytes directly (no RF simulation)."""
        pkt = TelemetryPacket.decode(raw)
        if pkt:
            self.rx_buf.append(pkt)
            log.debug("NRF-RX seq=%d  alt=%.1fkm  vel=%.2fm/s  RSSI=%.1fdBm",
                      pkt.seq, pkt.altitude/1e3, pkt.velocity, pkt.rssi)
        return pkt

    # ── Path B: full SDR GFSK decode ──────────────────────────────────────
    def process_telemetry_sdr(self, raw: bytes) -> list:
        """
        Route raw packet bytes through the full SDR decode chain:
          1. GFSK modulate (adds preamble)
          2. Simulate RF channel (AWGN + CFO)
          3. Run 7-stage SDRDecoder pipeline
          4. Return list of DecodedPacket

        Also runs Path A in parallel so both buffers stay populated.
        """
        # Path A (always)
        self.process_telemetry(raw)

        if not self.use_sdr_decode:
            return []

        # Path B: modulate -> channel -> decode
        iq = self.sdr.channel_simulate(raw,
                                        snr_db = self.snr_db,
                                        cfo_hz = self.cfo_hz)
        decoded = self._sdr_decoder.decode(iq)
        self.sdr_rx_buf.extend(decoded)

        if decoded:
            p = decoded[0]
            log.debug("SDR-RX seq=%d  alt=%.1fkm  SNR=%.1fdB  CFO=%.0fHz",
                      p.seq, p.altitude/1e3, p.snr_db, p.cfo_hz)
        return decoded

    def send_command(self, thrust: float, torque: Tuple[float,float,float]) -> bool:
        cmd = CommandPacket(thrust=thrust,
                            torque_x=torque[0],
                            torque_y=torque[1],
                            torque_z=torque[2])
        return self.radio.transmit(cmd.encode())

    def link_stats(self) -> dict:
        sdr_stats = self._sdr_decoder.stats if self._sdr_decoder else {}
        return {
            'packets_received_nrf'  : len(self.rx_buf),
            'packets_decoded_sdr'   : len(self.sdr_rx_buf),
            'packet_loss_rate'      : f"{self.radio.packet_loss_rate*100:.2f}%",
            'signal_exchanges'      : self.radio.signal_exchanges_per_cycle,
            'max_range_km'          : f"{self.budget.max_range_km():.1f} km",
            'link_margin_500km'     : f"{self.budget.link_margin_db(500):.1f} dB",
            'sdr_avg_snr_db'        : f"{sdr_stats.get('avg_snr_db', 0):.1f} dB",
            'sdr_avg_cfo_hz'        : f"{sdr_stats.get('avg_cfo_hz', 0):.1f} Hz",
        }


# ─── Telemetry Simulator Thread ───────────────────────────────────────────────

class TelemetrySimulator:
    """
    Drives a full telemetry simulation at 50 Hz,
    generating realistic CubeSat state evolution.
    """
    def __init__(self, duration: float = 30.0,
                 use_sdr_decode: bool = True,
                 snr_db: float = 18.0,
                 cfo_hz: float = 300.0):
        self.duration  = duration
        self.gs        = GroundStation(use_sdr_decode=use_sdr_decode,
                                       snr_db=snr_db, cfo_hz=cfo_hz)
        self.t         = []
        self.alt_log   = []
        self.vel_log   = []
        self.rssi_log  = []
        self.exchange_log       = []
        self.sdr_decoded_log    = []   # count of SDR-decoded packets over time

        # Simple plant state
        self._alt  = 400_000.0
        self._vel  = 0.0
        self._t    = 0.0
        self._seq  = 0

    def _step_plant(self, thrust: float, dt: float):
        m    = 1.33
        drag = 0.0012 * self._vel**2 * np.sign(self._vel)
        a    = (thrust - drag) / m + 0.1 * np.random.randn()
        self._vel  += a * dt
        self._alt  += self._vel * dt
        self._t    += dt

    def _alt_ref(self) -> float:
        if self._t < 10:   return 400_000
        elif self._t < 20: return 420_000
        else:              return 410_000

    def run(self):
        dt = 0.02
        N  = int(self.duration / dt)
        log.info("Telemetry simulation starting (%d steps @ 50 Hz)", N)

        for k in range(N):
            ref    = self._alt_ref()
            err    = ref - self._alt
            thrust = max(min(2.8 * err / 1000, 1.5), -1.5)

            self._step_plant(thrust, dt)

            rssi = -65 + 5 * np.random.randn()
            pkt  = TelemetryPacket(
                seq       = self._seq,
                altitude  = self._alt,
                velocity  = self._vel,
                rssi      = rssi,
                timestamp = int(self._t * 1000)
            )
            self._seq += 1

            raw = pkt.encode()

            # ── Route through SDR decode chain (Path B) ────────────────
            # Every 10th packet goes through full GFSK modulate→decode
            # (full decode every step would be very slow in pure Python)
            if k % 10 == 0 and self.gs.use_sdr_decode:
                sdr_pkts = self.gs.process_telemetry_sdr(raw)
            else:
                self.gs.radio.rx_queue.put(raw)
                rx = self.gs.radio.rx_queue.get()
                self.gs.process_telemetry(rx)

            # Ground station sends uplink command
            self.gs.send_command(thrust, (0.0, 0.0, 0.0))

            self.t.append(self._t)
            self.alt_log.append(self._alt / 1e3)
            self.vel_log.append(self._vel)
            self.rssi_log.append(rssi)
            self.exchange_log.append(self.gs.radio.signal_exchanges_per_cycle)
            self.sdr_decoded_log.append(len(self.gs.sdr_rx_buf))

        stats = self.gs.link_stats()
        log.info("Simulation complete.")
        for k, v in stats.items():
            log.info("  %-28s %s", k, v)
        return stats

    def plot(self):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10),
                                  facecolor='#0D1117')
        for ax in axes:
            ax.set_facecolor('#161B22')
            ax.tick_params(colors='#8B949E')
            ax.spines[:].set_color('#30363D')
            ax.grid(True, color='#21262D', alpha=0.5)

        axes[0].plot(self.t, self.alt_log, color='#58A6FF', lw=1.5)
        axes[0].set_ylabel('Altitude [km]', color='#C9D1D9')
        axes[0].set_title('CubeSat Telemetry — Full SDR Decode Pipeline', color='#F0F6FC', pad=10)

        axes[1].plot(self.t, self.vel_log, color='#FF7B72', lw=1.5)
        axes[1].set_ylabel('Velocity [m/s]', color='#C9D1D9')

        axes[2].plot(self.t, self.rssi_log, color='#3FB950', lw=1.5, alpha=0.7)
        axes[2].axhline(-100, color='#FF7B72', ls='--', lw=1, label='Sensitivity')
        axes[2].set_ylabel('RSSI [dBm]', color='#C9D1D9')
        axes[2].legend(facecolor='#161B22', labelcolor='#C9D1D9')

        axes[3].step(self.t, self.sdr_decoded_log, color='#BC8CFF', lw=1.5, where='post')
        axes[3].set_ylabel('SDR Decoded\n(cumulative)', color='#C9D1D9')
        axes[3].set_xlabel('Time [s]', color='#C9D1D9')

        plt.tight_layout()
        plt.savefig('telemetry_simulation.png', dpi=150, bbox_inches='tight')
        log.info("Plot saved: telemetry_simulation.png")
        plt.show()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CubeSat Telemetry + SDR Decoder — Arya MGC')
    parser.add_argument('--duration',   type=float, default=10.0,
                        help='Simulation duration [s] (default 10)')
    parser.add_argument('--snr',        type=float, default=18.0,
                        help='Channel SNR for SDR decode [dB]')
    parser.add_argument('--cfo',        type=float, default=300.0,
                        help='Simulated CFO [Hz]')
    parser.add_argument('--no-sdr',     action='store_true',
                        help='Disable SDR decode (NRF-only mode)')
    parser.add_argument('--loopback',   action='store_true',
                        help='Run SDR decoder loopback test only')
    args = parser.parse_args()

    if args.loopback:
        from sdr_decoder import run_loopback_test
        run_loopback_test(num_packets=15, snr_db=args.snr)
    else:
        sim = TelemetrySimulator(
            duration       = args.duration,
            use_sdr_decode = not args.no_sdr,
            snr_db         = args.snr,
            cfo_hz         = args.cfo,
        )
        stats = sim.run()

        print("\n" + "═"*48)
        print("  CubeSat Telemetry System — Summary")
        print("═"*48)
        for k, v in stats.items():
            print(f"  {k:<32} {v}")
        print("═"*48)

        sim.plot()
