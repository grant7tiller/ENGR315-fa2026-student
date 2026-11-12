"""ENGR Circuits HW - Problem 6.12: v_C(t) for t >= 0 (source-free series RLC)."""
import numpy as np
import matplotlib.pyplot as plt

# Given values
V0, R1, R2, L, C = 12.0, 0.4, 1.2, 0.1, 0.4

# Initial conditions (switch closed long before t = 0: L = short, C = open)
vC0 = V0 * R2 / (R1 + R2)   # 9 V
iL0 = 0.0                   # C blocks DC, so no current in L

# t > 0: R2, L, C form a source-free series RLC loop
alpha = R2 / (2 * L)                 # 6 Np/s
w0 = 1 / np.sqrt(L * C)              # 5 rad/s  -> alpha > w0: overdamped
s1 = -alpha + np.sqrt(alpha**2 - w0**2)
s2 = -alpha - np.sqrt(alpha**2 - w0**2)
vC_inf = 0.0

# vC(0) = A1 + A2;  dvC/dt(0) = iL(0)/C = s1*A1 + s2*A2
dv0 = iL0 / C
A1 = (dv0 - s2 * (vC0 - vC_inf)) / (s1 - s2)
A2 = (vC0 - vC_inf) - A1

print(f"alpha={alpha:.3f}, w0={w0:.3f}, s1={s1:.4f}, s2={s2:.4f}")
print(f"A1={A1:.4f}, A2={A2:.4f}")

t = np.linspace(0, 3, 1000)
vC = vC_inf + A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
t_ss = 5 / abs(s1)                   # ~5 time constants of the slow root

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.axvspan(0, t_ss, color="#2a6fdb", alpha=0.06)
ax.plot(t, vC, color="#2a6fdb", lw=2, label=r"$v_C(t)$")
ax.axhline(vC_inf, color="#666", lw=1.2, ls="--", label=r"Steady state $v_C(\infty)=0$ V")
ax.axvline(t_ss, color="#999", lw=1, ls=":")
ax.plot(0, vC0, "o", color="#2a6fdb", ms=6)
ax.annotate(r"$v_C(0)=9$ V, slope $=0$", (0, vC0), (0.35, 9.4), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#444"))
ax.text(t_ss / 2, 6.5, "Transient", ha="center", fontsize=10, color="#333")
ax.text((t_ss + 3) / 2, 1.2, "Steady state", ha="center", fontsize=10, color="#333")
ax.text(t_ss + 0.03, 4.5, rf"$5/|s_1|\approx{t_ss:.2f}$ s", fontsize=9, color="#555")
ax.set_title(r"Problem 6.12: $v_C(t)=12.64e^{-2.68t}-3.64e^{-9.32t}$ V  (overdamped)", fontsize=11)
ax.set_xlabel("t (s)")
ax.set_ylabel(r"$v_C$ (V)")
ax.set_xlim(0, 3); ax.set_ylim(-0.5, 10.5)
ax.grid(alpha=0.25)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P6_12_vC_plot.png")
fig.savefig(out)
print("Plot saved to:", out)
plt.show()