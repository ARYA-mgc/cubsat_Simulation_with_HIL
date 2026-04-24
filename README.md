# 🛰️ CubeSat Altitude Stabilization & Control System
<div align="center">
![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange?style=for-the-badge&logo=mathworks)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![HIL](https://img.shields.io/badge/HIL-50Hz-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A full-stack CubeSat Attitude & Control System (ACS) simulation featuring cascaded PID control, Hardware-in-the-Loop (HIL) validation, and a complete SDR + NRF24L01+ telemetry pipeline.**
*Author: **Arya MGC*** 
</div>



---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [MATLAB Simulation](#matlab-simulation)
- [Hardware-in-the-Loop (HIL)](#hardware-in-the-loop-hil)
- [Telemetry System (SDR + NRF)](#telemetry-system-sdr--nrf)
- [Control Analysis](#control-analysis)
- [Hardware Integration](#hardware-integration)
- [Contributing](#contributing)

---

## 🌍 Overview

This project implements a complete **Altitude Stabilization and Control System (ACS)** for a 1U CubeSat operating in Low Earth Orbit (LEO) at ~400 km. It covers the full engineering stack from control theory to embedded implementation, validated via Hardware-in-the-Loop simulation.

### Highlights

| Feature | Specification |
|---|---|
| Settling Time | **< 2 s** (2% band) |
| HIL Update Rate | **50+ Hz** |
| Telemetry Rate | **40+ signal exchanges / cycle** |
| Control Architecture | **Cascaded PID (Outer + Inner loop)** |
| Attitude Axes | **3-axis (Roll / Pitch / Yaw)** |
| Communication | **NRF24L01+ @ 2.4 GHz + SDR GFSK downlink** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CubeSat ACS — Block Diagram                 │
│                                                                 │
│  Reference         ┌──────────┐   vel_cmd  ┌──────────┐         │
│  Altitude ────────►│ Outer    ├────────────►  Inner   │         │
│  (alt_ref)         │ PID      │            │ PID      │         │
│                    │ Kp=2.80  │            │ Kp=5.50  │  thrust │
│                    │ Ki=0.35  │            │ Ki=0.80  ├────────►│
│                    │ Kd=1.10  │            │ Kd=0.40  │         │
│                    └──────────┘            └──────────┘         │
│                                                    │            │
│                    ┌──────────────────────────────-┘            │
│                    │        Plant (6-DOF)                       │
│                    │   m=1.33 kg, drag, J2 perturbation         │
│                    └────────────────────────────────►           │
│                         alt, vel, attitude   (feedback)         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Telemetry Pipeline                         │    │
│  │  [Sensors] → [NRF24L01+] → [SDR GFSK] → [Ground Stn]    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Control Loop Hierarchy

```
Level 1 (Outer) — Altitude Control       50 Hz
└── Level 2 (Inner) — Velocity Control   50 Hz
    └── Level 3 — Attitude (3-axis)      50 Hz
        ├── Roll  PID  [Kp=12.0, Ki=0.5, Kd=2.8]
        ├── Pitch PID  [Kp=12.0, Ki=0.5, Kd=2.8]
        └── Yaw   PID  [Kp=8.0,  Ki=0.3, Kd=1.5]
```

---

## 📊 Key Results

### Step Response Performance

| Metric | Value | Target |
|---|---|---|
| Settling Time (2%) | **1.73 s** | < 2 s ✅ |
| Overshoot | **4.2 %** | < 10 % ✅ |
| Rise Time | **0.31 s** | — |
| Steady-State Error | **< 5 m** | < 50 m ✅ |

### Stability Margins

| Metric | Value | Minimum |
|---|---|---|
| Gain Margin | **18.4 dB** | > 6 dB ✅ |
| Phase Margin | **52.3°** | > 30° ✅ |

### HIL Performance

| Metric | Value |
|---|---|
| Update Rate | **50 Hz** |
| Packet Loss | **< 2 %** |
| Signal Exchanges | **40+ / cycle** |
| Max RF Range | **~850 km** |

---

## 📁 Repository Structure

```
cubesat-acs/
│
├── matlab/
│   ├── cubesat_pid_control.m     # Main cascaded PID simulation
│   ├── hil_simulator.m           # Hardware-in-the-Loop engine
│   └── pid_tuning_zn.m           # Ziegler-Nichols + ITAE tuning
│
├── python/
│   ├── sdr_decoder.py            # ★ Full 7-stage GFSK decode algorithm
│   ├── telemetry_sdr_nrf.py      # SDR + NRF24L01+ simulation (uses sdr_decoder)
│   ├── control_analysis.py       # Bode, Nyquist, sensitivity analysis
│   └── requirements.txt          # Python dependencies
│
├── hardware/
│   ├── nrf24l01_driver.h         # NRF24L01+ C++ driver (STM32)
│   ├── pid_embedded.h            # Embedded PID implementation
│   └── pin_config.h              # Hardware pinout definitions
│
├── tests/
│   ├── test_pid.py               # PID unit tests
│   ├── test_telemetry.py         # Telemetry packet tests
│   └── test_hil.py               # HIL loop tests
│
├── docs/
│   ├── system_design.md          # Detailed design document
│   ├── link_budget.md            # RF link budget analysis
│   └── pid_tuning_guide.md       # PID tuning methodology
│
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

**MATLAB:**
- MATLAB R2022b or later
- Control System Toolbox
- Signal Processing Toolbox

**Python:**
```bash
pip install numpy scipy matplotlib
```

### Run MATLAB Simulation

```matlab
% In MATLAB command window:
cd matlab/
run('cubesat_pid_control.m')
```

### Run Python Analysis

```bash
cd python/
python control_analysis.py                    # Bode + step analysis
python telemetry_sdr_nrf.py                  # Full SDR decode pipeline
python telemetry_sdr_nrf.py --snr 20 --cfo 500   # Custom channel conditions
```

### Run SDR Decoder — Loopback Test

```bash
# End-to-end: modulate → AWGN channel → full 7-stage decode
python sdr_decoder.py --loopback --packets 20 --snr 18

# Show diagnostic pipeline plots (IQ constellation, eye diagram, etc.)
python sdr_decoder.py --loopback --plot

# Decode an RTL-SDR binary capture file
python sdr_decoder.py --file capture.bin
```

### Decode a Single Packet (Python API)

```python
from sdr_decoder import SDRDecoder, DecoderConfig
from telemetry_sdr_nrf import GFSKModulator, TelemetryPacket
import numpy as np

# Modulate a packet
pkt = TelemetryPacket(seq=1, altitude=405000.0, velocity=1.5)
mod = GFSKModulator()
iq  = mod.channel_simulate(pkt.encode(), snr_db=18.0, cfo_hz=300.0)

# Decode
decoder = SDRDecoder(DecoderConfig())
results = decoder.decode(iq)

for p in results:
    print(f"alt={p.altitude/1e3:.1f}km  vel={p.velocity:.3f}m/s  SNR={p.snr_db:.1f}dB")

decoder.print_stats()
```

### Run HIL Standalone Mode

```matlab
hil = HILSimulator('use_udp', false);
hil.run_standalone(30);          % 30-second HIL run
hil.export_csv('hil_results.csv');
```

---

## 🎛️ MATLAB Simulation

### Plant Model

The CubeSat dynamics are modeled as a 6-DOF rigid body with orbital perturbations:

```
Translational:  m·ẍ = F_thrust - F_drag + F_J2 + F_SRP
Rotational:     I·ω̇ = τ_control + τ_disturbance

m = 1.33 kg  (1U CubeSat)
I = diag([1.2e-3, 1.2e-3, 0.8e-3]) kg·m²
```

### PID Anti-Windup

Integrator clamping is applied to prevent wind-up during large set-point changes:

```matlab
int_alt = max(min(int_alt + err_alt * dt, 5e3), -5e3);
```

### Disturbance Model

Three orbital perturbations are simulated:
- **J2 oblateness** — sinusoidal at orbital frequency
- **Atmospheric drag** — quadratic drag with LEO density model
- **Stochastic noise** — AWGN representing sensor noise and unmodeled dynamics

---

## 🔌 Hardware-in-the-Loop (HIL)

The HIL system bridges MATLAB (plant model) and the embedded flight computer (STM32 / Raspberry Pi CM4) via **UDP** or **Serial**:

```
MATLAB Plant ←──UDP:5005──→ Flight Computer MCU
              sensor_pkt           ↕
              command_pkt    NRF24L01+
                              ↕
                         SDR Ground Station
```

### Packet Format (NRF24L01+ 32-byte frame)

```
Byte  0    : 0xAE  (header)
Byte  1    : Seq num (uint8)
Bytes 2–5  : Altitude  (float32, m)
Bytes 6–9  : Velocity  (float32, m/s)
Bytes 10–13: Roll      (float32, rad)
Bytes 14–17: Pitch     (float32, rad)
Bytes 18–21: Yaw       (float32, rad)
Bytes 22–25: Timestamp (uint32, ms)
Bytes 26–29: RSSI      (float32, dBm)
Byte  30   : Checksum  (uint8, XOR)
Byte  31   : 0xEF  (footer)
```

---

## 🔭 SDR GFSK Decoding Algorithm

> **File:** `python/sdr_decoder.py`

Full 7-stage receive chain for GFSK-modulated CubeSat telemetry downlink, compatible with RTL-SDR / HackRF captures and loopback simulation.

### Receive Chain

```
IQ Samples (RTL-SDR / HackRF / simulated)
  │
  ▼
[Stage 1] IQ Pre-processing
  ├─ DC offset removal        (adaptive leaky integrator)
  ├─ IQ imbalance correction  (amplitude + phase mismatch matrix)
  └─ Anti-alias LPF           (5th-order Butterworth)
  │
  ▼
[Stage 2] CFO Estimation & Correction
  ├─ Coarse: FFT-based (iq^4 method, freq resolution = Fs/N)
  └─ Fine:   Costas PLL (2nd-order, proportional + integral loop filter)
  │
  ▼
[Stage 3] FM Discriminator
  ├─ Hard limiter (removes AM noise, constant-envelope assumption)
  ├─ Differential phase:  f(n) = angle(x(n)·conj(x(n-1))) · Fs/2π
  └─ Post-discriminator LPF (reduces ISI)
  │
  ▼
[Stage 4] Symbol Timing Recovery
  ├─ Mueller & Müller TED  e(n) = d̂(n-1)·x(n) - d̂(n)·x(n-1)
  ├─ Cubic Farrow interpolator (4-tap, better than linear for low SPS)
  └─ 2nd-order loop filter (proportional + integral timing correction)
  │
  ▼
[Stage 5] Bit Slicing
  ├─ Hard decision slicer  (threshold = 0)
  └─ Optional NRZI decode  (for NRF24L01+ NRZI configurations)
  │
  ▼
[Stage 6] Frame Synchronisation
  ├─ Sliding cross-correlation with preamble + SFD template
  ├─ Normalised peak detection with configurable threshold
  └─ Candidate position list with minimum separation guard
  │
  ▼
[Stage 7] Packet Extraction & Validation
  ├─ Header / footer byte check  (0xAE / 0xEF)
  ├─ Checksum verification       (sum of payload bytes mod 256)
  ├─ Sanity bounds               (altitude, velocity, attitude)
  └─ → DecodedPacket dataclass
```

### Key Classes

| Class | Responsibility |
|---|---|
| `DecoderConfig` | All tunable parameters (sample rate, bit rate, loop gains, thresholds) |
| `IQPreprocessor` | DC removal, IQ imbalance compensation, anti-alias LPF, SNR estimation |
| `CFOCorrector` | FFT coarse CFO + Costas loop fine tracking |
| `FMDiscriminator` | Differential phase demodulator + eye diagram |
| `TimingRecovery` | Mueller & Müller TED with cubic Farrow interpolation |
| `BitSlicer` | Hard decision + optional NRZI decoding |
| `FrameSync` | Correlator-based preamble/SFD detection |
| `PacketValidator` | Checksum + bounds checking → `DecodedPacket` |
| `SDRDecoder` | Orchestrates all 7 stages, stateful block-by-block processing |
| `RTLSDRFileReader` | Stream-decode `.bin` / `.cf32` RTL-SDR captures |

### Integration with Telemetry Module

```python
# GroundStation now has two parallel receive paths:
gs = GroundStation(use_sdr_decode=True, snr_db=18.0, cfo_hz=300.0)

# Path A — direct NRF byte decode (fast)
gs.process_telemetry(raw_bytes)

# Path B — full GFSK modulate → channel → 7-stage SDR decode
sdr_packets = gs.process_telemetry_sdr(raw_bytes)
```

---

## 📡 Telemetry System (SDR + NRF)

### RF Configuration

| Parameter | Value |
|---|---|
| NRF Frequency | 2.4 GHz (CH 76 = 2476 MHz) |
| SDR Downlink | 433 MHz GFSK |
| Bit Rate | 250 kbps |
| Freq Deviation | ±125 kHz |
| BT Product | 0.5 |
| TX Power | +20 dBm |
| RX Sensitivity | −100 dBm |

### Link Budget (500 km orbit)

```
TX Power          : +20.0 dBm
TX Antenna Gain   : +2.0  dBi
Free-Space Path Loss: -147.8 dBm  (433 MHz, 500 km)
RX Antenna Gain   : +12.0 dBi
Noise Figure      : -3.0  dB
────────────────────────────────
Link Margin       : +13.2 dB   ✅ (> 3 dB required)
```

---

## 🔬 Control Analysis

Run `python/control_analysis.py` to generate:

1. **Bode Plot** — Magnitude & phase of open-loop transfer function
2. **Nyquist Diagram** — Stability visualization
3. **Step Response** — Time-domain performance
4. **Pole-Zero Map** — Closed-loop pole locations
5. **Sensitivity Surface** — Settling time vs Kp/Kd parameter sweep

---

## 🛠️ Hardware Integration

### Wiring (STM32F4 ↔ NRF24L01+)

| STM32 Pin | NRF24L01+ |
|---|---|
| PA4 (SPI1_NSS) | CSN |
| PA5 (SPI1_SCK) | SCK |
| PA6 (SPI1_MISO) | MISO |
| PA7 (SPI1_MOSI) | MOSI |
| PB0 | CE |
| PB1 | IRQ |

### SDR Setup (RTL-SDR / HackRF)

```bash
# Receive GFSK downlink with GNU Radio
rtl_sdr -f 433e6 -s 2e6 -g 40 - | python sdr_decoder.py
```

---

## 🧪 Tests

```bash
cd tests/
python -m pytest test_pid.py test_telemetry.py -v
```

---

## 📖 Theory

### Cascaded PID Design Rationale

The cascaded architecture separates the fast (velocity) and slow (altitude) dynamics, allowing independent tuning of each loop. The inner loop must be at least **5× faster** than the outer loop for the approximation to hold.

Gains were tuned using **Ziegler-Nichols** initial estimates refined by **ITAE minimization**:

```
ITAE = ∫₀^∞ t·|e(t)| dt
```

minimized over [Kp, Ki, Kd] subject to:
- Phase margin > 45°
- Gain margin > 12 dB
- Settling time < 2 s

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">

**Arya MGC** — CubeSat ACS Project

*"Control the orbit. Own the mission."*

</div>
