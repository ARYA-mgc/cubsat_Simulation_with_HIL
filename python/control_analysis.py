"""
CubeSat ACS — PID Analysis & Frequency Response
Author: Arya MGC

Computes:
  - Bode plots for cascaded PID loops
  - Gain/phase margin analysis
  - Step response metrics (settling time, overshoot, ITAE)
  - Root locus visualization
  - PID parameter sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    name: str = "PID"

    def tf(self, s: np.ndarray) -> np.ndarray:
        """PID transfer function: C(s) = Kp + Ki/s + Kd*s"""
        return self.kp + self.ki / s + self.kd * s

    def as_scipy_tf(self) -> signal.TransferFunction:
        """Return scipy TransferFunction for PID"""
        # C(s) = (Kd·s² + Kp·s + Ki) / s
        num = [self.kd, self.kp, self.ki]
        den = [1, 0]
        return signal.TransferFunction(num, den)


@dataclass
class PlantModel:
    """Second-order plant model for altitude channel: P(s) = 1/(m·s²)"""
    mass: float = 1.33
    drag: float = 0.0012

    def tf(self) -> signal.TransferFunction:
        # Linearized: P(s) ≈ 1 / (m·s² + b·s)
        return signal.TransferFunction([1], [self.mass, self.drag, 0])


class ControlAnalysis:

    def __init__(self):
        # Tuned PID gains
        self.outer = PIDGains(kp=2.80, ki=0.35, kd=1.10, name="Altitude (Outer)")
        self.inner = PIDGains(kp=5.50, ki=0.80, kd=0.40, name="Velocity (Inner)")
        self.plant = PlantModel()

    def open_loop_tf(self) -> signal.TransferFunction:
        """Cascaded PID open-loop transfer function"""
        pid_o = self.outer.as_scipy_tf()
        pid_i = self.inner.as_scipy_tf()
        p     = self.plant.tf()
        ol    = signal.series(pid_o, signal.series(pid_i, p))
        return ol

    def closed_loop_tf(self) -> signal.TransferFunction:
        """Unity feedback closed-loop"""
        ol = self.open_loop_tf()
        return signal.feedback(ol, sign=-1)

    def stability_margins(self) -> Dict[str, float]:
        """Compute gain margin, phase margin"""
        ol  = self.open_loop_tf()
        w   = np.logspace(-3, 4, 10000)
        _, H = signal.freqs(ol.num, ol.den, w)

        # Phase margin: phase at |H| = 1
        mag   = np.abs(H)
        phase = np.angle(H, deg=True)
        idx_gc = np.argmin(np.abs(mag - 1.0))
        pm     = 180 + phase[idx_gc]

        # Gain margin: gain at phase = -180°
        idx_pc = np.argmin(np.abs(phase + 180))
        gm_db  = -20 * np.log10(mag[idx_pc]) if mag[idx_pc] > 0 else np.inf

        return {
            'gain_margin_db'  : gm_db,
            'phase_margin_deg': pm,
            'gain_crossover_hz': w[idx_gc] / (2*np.pi),
            'phase_crossover_hz': w[idx_pc] / (2*np.pi),
        }

    def step_metrics(self) -> Dict[str, float]:
        """Compute step response performance indices"""
        cl  = self.closed_loop_tf()
        t   = np.linspace(0, 10, 5000)
        _, y = signal.step(cl, T=t)

        # Settling time (2% band)
        ss  = y[-1]
        tol = 0.02 * abs(ss)
        settled = np.where(np.abs(y - ss) > tol)[0]
        t_s = t[settled[-1]] if len(settled) > 0 else 0.0

        # Overshoot
        os  = (np.max(y) - ss) / ss * 100 if ss != 0 else 0.0

        # Rise time (10%–90%)
        y10 = np.where(y >= 0.1*ss)[0]
        y90 = np.where(y >= 0.9*ss)[0]
        t_r = (t[y90[0]] - t[y10[0]]) if len(y10) and len(y90) else 0.0

        # ITAE index
        itae = np.trapz(t * np.abs(1 - y), t)

        return {
            'settling_time_s' : t_s,
            'overshoot_pct'   : os,
            'rise_time_s'     : t_r,
            'itae_index'      : itae,
            'steady_state_err': abs(1 - ss),
        }

    def sensitivity_analysis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sweep Kp and compute settling time for sensitivity surface"""
        kp_range = np.linspace(0.5, 6.0, 20)
        kd_range = np.linspace(0.1, 2.5, 20)
        KP, KD   = np.meshgrid(kp_range, kd_range)
        T_settle = np.zeros_like(KP)

        t  = np.linspace(0, 15, 3000)
        p  = self.plant.tf()

        for i in range(len(kp_range)):
            for j in range(len(kd_range)):
                pid_o = PIDGains(KP[j,i], 0.35, KD[j,i]).as_scipy_tf()
                pid_i = self.inner.as_scipy_tf()
                ol    = signal.series(pid_o, signal.series(pid_i, p))
                cl    = signal.feedback(ol)
                try:
                    _, y = signal.step(cl, T=t)
                    ss   = y[-1]
                    tol  = 0.02 * abs(ss) if ss != 0 else 0.02
                    sett = np.where(np.abs(y - ss) > tol)[0]
                    T_settle[j,i] = t[sett[-1]] if len(sett) > 0 else 0.01
                except Exception:
                    T_settle[j,i] = 15.0   # Unstable
        return KP, KD, T_settle

    def plot_all(self):
        fig = plt.figure(figsize=(16, 12), facecolor='#0D1117')
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.40)

        ax_bode_mag   = fig.add_subplot(gs[0, 0:2])
        ax_bode_phase = fig.add_subplot(gs[1, 0:2])
        ax_step       = fig.add_subplot(gs[0, 2])
        ax_sensitivity= fig.add_subplot(gs[1, 2])
        ax_nyquist    = fig.add_subplot(gs[2, 0])
        ax_pzmap      = fig.add_subplot(gs[2, 1])
        ax_metrics    = fig.add_subplot(gs[2, 2])

        dark = '#0D1117'
        panel= '#161B22'
        txt  = '#C9D1D9'
        acc1 = '#58A6FF'
        acc2 = '#FF7B72'
        acc3 = '#3FB950'
        grid_c = '#21262D'

        for ax in fig.get_axes():
            ax.set_facecolor(panel)
            ax.tick_params(colors='#8B949E', labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#30363D')
            ax.grid(True, color=grid_c, alpha=0.6, linewidth=0.5)

        # ── Bode plot ─────────────────────────────────────────────────────
        ol  = self.open_loop_tf()
        w   = np.logspace(-2, 4, 5000)
        _, H = signal.freqs(ol.num, ol.den, w)
        f   = w / (2 * np.pi)

        ax_bode_mag.semilogx(f, 20*np.log10(np.abs(H)+1e-12), color=acc1, lw=1.5)
        ax_bode_mag.axhline(0, color=acc2, ls='--', lw=1, alpha=0.8, label='0 dB')
        ax_bode_mag.set_ylabel('Magnitude [dB]', color=txt, fontsize=8)
        ax_bode_mag.set_title('Bode Plot — Open Loop', color=txt, fontsize=9, fontweight='bold')
        ax_bode_mag.legend(fontsize=7, facecolor=panel, labelcolor=txt)

        ax_bode_phase.semilogx(f, np.angle(H, deg=True), color=acc3, lw=1.5)
        ax_bode_phase.axhline(-180, color=acc2, ls='--', lw=1, alpha=0.8, label='-180°')
        ax_bode_phase.set_ylabel('Phase [deg]', color=txt, fontsize=8)
        ax_bode_phase.set_xlabel('Frequency [Hz]', color=txt, fontsize=8)
        ax_bode_phase.legend(fontsize=7, facecolor=panel, labelcolor=txt)

        # ── Step Response ────────────────────────────────────────────────
        cl  = self.closed_loop_tf()
        t   = np.linspace(0, 5, 2000)
        _, y = signal.step(cl, T=t)
        ax_step.plot(t, y,   color=acc1, lw=1.5, label='Response')
        ax_step.plot(t, np.ones_like(t)*y[-1], color=acc2, ls='--', lw=1, label='Ref')
        ax_step.set_title('Step Response', color=txt, fontsize=9, fontweight='bold')
        ax_step.set_xlabel('Time [s]', color=txt, fontsize=8)
        ax_step.legend(fontsize=7, facecolor=panel, labelcolor=txt)

        # ── Nyquist ──────────────────────────────────────────────────────
        ax_nyquist.plot(H.real, H.imag, color=acc1, lw=1.2)
        ax_nyquist.plot(-1, 0, 'r+', ms=10)
        ax_nyquist.set_title('Nyquist', color=txt, fontsize=9, fontweight='bold')
        ax_nyquist.set_xlabel('Real', color=txt, fontsize=8)
        ax_nyquist.set_ylabel('Imag', color=txt, fontsize=8)

        # ── Pole-Zero Map ────────────────────────────────────────────────
        poles = cl.poles
        zeros = cl.zeros
        ax_pzmap.axhline(0, color='#30363D', lw=0.8)
        ax_pzmap.axvline(0, color='#30363D', lw=0.8)
        ax_pzmap.plot(poles.real, poles.imag, 'x', color=acc2, ms=8, mew=2)
        ax_pzmap.plot(zeros.real, zeros.imag, 'o', color=acc3, ms=6, mew=1.5, fillstyle='none')
        ax_pzmap.set_title('Pole-Zero Map', color=txt, fontsize=9, fontweight='bold')
        ax_pzmap.set_xlabel('Real', color=txt, fontsize=8)

        # ── Sensitivity surface ──────────────────────────────────────────
        KP, KD, T_s = self.sensitivity_analysis()
        cs = ax_sensitivity.contourf(KP, KD, np.clip(T_s, 0, 5),
                                      levels=20, cmap='RdYlGn_r')
        ax_sensitivity.set_title('Settling Time Sensitivity', color=txt, fontsize=9, fontweight='bold')
        ax_sensitivity.set_xlabel('Kp', color=txt, fontsize=8)
        ax_sensitivity.set_ylabel('Kd', color=txt, fontsize=8)
        plt.colorbar(cs, ax=ax_sensitivity).ax.tick_params(colors=txt)

        # ── Metrics table ────────────────────────────────────────────────
        m   = self.step_metrics()
        sm  = self.stability_margins()
        ax_metrics.axis('off')
        rows = [
            ['Settling Time',  f"{m['settling_time_s']:.3f} s"],
            ['Overshoot',      f"{m['overshoot_pct']:.2f} %"],
            ['Rise Time',      f"{m['rise_time_s']:.3f} s"],
            ['ITAE Index',     f"{m['itae_index']:.4f}"],
            ['Gain Margin',    f"{sm['gain_margin_db']:.1f} dB"],
            ['Phase Margin',   f"{sm['phase_margin_deg']:.1f}°"],
        ]
        tbl = ax_metrics.table(cellText=rows,
                                colLabels=['Metric', 'Value'],
                                cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_facecolor('#21262D' if r == 0 else panel)
            cell.set_text_props(color=acc1 if c==1 and r>0 else txt)
            cell.set_edgecolor('#30363D')
        ax_metrics.set_title('Performance Summary', color=txt, fontsize=9, fontweight='bold')

        fig.suptitle('CubeSat ACS — Control Analysis Dashboard  |  Arya MGC',
                     color='#F0F6FC', fontsize=12, fontweight='bold', y=0.98)
        plt.savefig('control_analysis.png', dpi=150, bbox_inches='tight',
                    facecolor=dark)
        print("Saved: control_analysis.png")
        plt.show()


if __name__ == '__main__':
    ca = ControlAnalysis()

    print("═"*48)
    print("  CubeSat ACS — Control Analysis")
    print("═"*48)

    metrics = ca.step_metrics()
    margins = ca.stability_margins()

    for k, v in {**metrics, **margins}.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print("═"*48)
    ca.plot_all()
