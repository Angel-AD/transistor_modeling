import math
import torch
import torch.nn.functional as F

def classic_angelov(X, Ipk, Vpk, P1, P2, P3, alpha, lambda_):
    Vgs, Vds = X[:, 0], X[:, 1]
    psi = P1*(Vgs - Vpk) + P2*(Vgs - Vpk)**2 + P3*(Vgs - Vpk)**3
    return Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha * Vds) * (1 + lambda_ * Vds)

def angelov_6_term(X, Ipk, Vpk, P1, P2, P3, P4, P5, P6, alpha, lambda_):
    """Extended Angelov with 6 polynomial terms for tighter gm fitting."""
    Vgs, Vds = X[:, 0], X[:, 1]
    dv = Vgs - Vpk
    
    psi = P1*dv + P2*(dv**2) + P3*(dv**3) + P4*(dv**4) + P5*(dv**5) + P6*(dv**6)
    
    return Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha * Vds) * (1 + lambda_ * Vds)


def angelov_9_term(X, Ipk, Vpk, P1, P2, P3, P4, P5, P6, P7, P8, P9, alpha, lambda_):
    """Ultra-high precision Angelov. Requires small LRs on higher terms to prevent NaN."""
    Vgs, Vds = X[:, 0], X[:, 1]
    dv = Vgs - Vpk
    
    psi = (P1*dv + P2*(dv**2) + P3*(dv**3) + 
           P4*(dv**4) + P5*(dv**5) + P6*(dv**6) + 
           P7*(dv**7) + P8*(dv**8) + P9*(dv**9))
           
    return Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha * Vds) * (1 + lambda_ * Vds)


def angelov_dual_peak(X, Ipk1, Vpk1, P1_1, P2_1, P3_1, Ipk2, Vpk2, P1_2, alpha, lambda_):
    """
    Dual-channel Angelov for GaN/GaAs devices with gm1 dispersion/double-humps.
    gm1 loss is highly effective here at locating the secondary conduction path (Vpk2).
    """
    Vgs, Vds = X[:, 0], X[:, 1]
    
    # Main conduction channel
    psi1 = P1_1*(Vgs - Vpk1) + P2_1*(Vgs - Vpk1)**2 + P3_1*(Vgs - Vpk1)**3
    I_ch1 = Ipk1 * (1 + torch.tanh(psi1))
    
    # Secondary conduction channel (parasitic/surface), usually needs fewer polynomial terms
    psi2 = P1_2*(Vgs - Vpk2)
    I_ch2 = Ipk2 * (1 + torch.tanh(psi2))
    
    # Combined current
    return (I_ch1 + I_ch2) * torch.tanh(alpha * Vds) * (1 + lambda_ * Vds)


def modern_angelov(X, Ipk, Vpk, P1, P2, P3, P4, alpha, lambda_, alpha_s):
    Vgs, Vds = X[:, 0], X[:, 1]
    psi = P1*(Vgs - Vpk) + P2*(Vgs - Vpk)**2 + P3*(Vgs - Vpk)**3 + P4*(Vgs - Vpk)**4
    alpha_eff = alpha + alpha_s * (1 + torch.tanh(psi))
    return Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha_eff * Vds) * (1 + lambda_ * Vds)

def mod1_angelov(X, Ipk, Ipk1, P1, P21, P22, P31, P32, Vpks, deltVpks, alphaR, alphaS, lambda_, n, Vgsf, deltP1, deltP21, deltP22, deltP31, deltP32):
    Vgs, Vds = X[:, 0], X[:, 1]
    Vpk = Vpks - deltVpks + deltVpks * torch.tanh(alphaS * Vds)
    Vgsp = Vgs - Vpk
    
    safe_n = n + 1e-12 
    x = n * Vgsp
    Vgspa = (1.0 / safe_n) * (torch.logaddexp(x, -x) - torch.log(torch.tensor(2.0))) 
    
    Veffp1 = 0.5 * (Vgsp - Vgspa)
    Veffp2 = 0.5 * (Vgsf + Vgspa)
    
    P1m = P1 * (1 + deltP1) * (1 + torch.tanh(alphaS * Vds))
    P21m = P21 * (1 + deltP21) * (1 + torch.tanh(alphaS * Vds))
    P22m = P22 * (1 + deltP22) * (1 + torch.tanh(alphaS * Vds))
    P31m = P31 * (1 + deltP31) * (1 + torch.tanh(alphaS * Vds))
    P32m = P32 * (1 + deltP32) * (1 + torch.tanh(alphaS * Vds))
    
    P111 = P1 * Ipk / (Ipk1 + 1e-12) 
    
    ph1 = P1m * Veffp1 + P21m * Veffp1**2 + P31m * Veffp1**3
    ph2 = P111 * Veffp2 + P22m * Veffp2**2 + P32m * Veffp2**3
    
    Ids1 = Ipk * (1 + torch.tanh(ph1)) + Ipk1 * torch.tanh(ph2)
    alpha_eff = alphaR + alphaS * (1 + torch.tanh(ph1))
    Ids2 = torch.tanh(alpha_eff * Vds) * (1 + lambda_ * Vds)
    return Ids1 * Ids2

def mod2_angelov(X, Ipk, Mipk1, qm, Vgsm, Vpk1, Vpk2, Vpk3, alpha, K10, K11, alpha1, K20, K21, alpha2, K30, K31, alpha3):
    Vgs, Vds = X[:, 0], X[:, 1]
    Vgsp1, Vgsp2, Vgsp3 = Vgs - Vpk1, Vgs - Vpk2, Vgs - Vpk3
    
    Pk1 = K10 + (K10 + K11 * Vds) * torch.tanh(alpha1 * Vds)
    Pk2 = K20 + (K20 + K21 * Vds) * torch.tanh(alpha2 * Vds)
    Pk3 = K30 + (K30 + K31 * Vds) * torch.tanh(alpha3 * Vds)
    
    PhiP = Pk1 * Vgsp1 + Pk2 * (Vgsp2)**2 + Pk3 * (Vgsp3)**3
    Mipk = 1 + 0.5 * (1 + Mipk1) * torch.tanh(qm * (Vgs - Vgsm))
    
    Ids1 = Ipk * (1 + Mipk * torch.tanh(PhiP))
    Ids2 = torch.tanh(alpha * Vds)
    return Ids1 * Ids2

def mod3_angelov(X, I0, I1, I2, I3, Mpk0, MpkA, Vgm0, Vpk, P1m, P2, P3, Pz0, Pz1, alphaZ, alpha):
    Vgs, Vds = X[:, 0], X[:, 1]
    Ipk = I0 + I1 * Vds + I2 * Vds**2 + I3 * Vds**3
    Zm = (Pz0 + Pz1 * Vds) * torch.tanh(alphaZ * Vds) + Pz0
    PhiM = Zm * (Vgs - Vgm0)
    
    Mpk = Mpk0 + MpkA * torch.tanh(PhiM)
    Vgsp = Vgs - Vpk
    PhiP = P1m * Vgsp + P2 * (Vgsp)**2 + P3 * (Vgsp)**3
    
    Ids1 = Ipk * (1 + Mpk * torch.tanh(PhiP))
    # safe_alpha_vds = torch.clamp(alpha * Vds, -20.0, 20.0)
    # Ids2 = torch.tanh(torch.sinh(safe_alpha_vds))
    Ids2 = torch.tanh(alpha * Vds)
    return Ids1 * Ids2

def angelov_with_correction(X, Ipk, Vpk, P1, P2, P3, alpha, lambda_, C_alpha, C_beta, C_vshift):
    Vgs, Vds = X[:, 0], X[:, 1]
    
    # 1. Base Angelov Core
    psi = P1*(Vgs - Vpk) + P2*(Vgs - Vpk)**2 + P3*(Vgs - Vpk)**3
    base_ids = Ipk * (1 + torch.tanh(psi)) * torch.tanh(alpha * Vds) * (1 + lambda_ * Vds)
    
    # 2. Asymmetric Correction Term (acts as a shock-absorber for gm transitions)
    # C_alpha: Amplitude of correction
    # C_beta: Sharpness of the transition
    # C_vshift: Where the correction activates on the Vgs axis
    correction = C_alpha * F.softplus(C_beta * (Vgs - C_vshift))
    
    return base_ids + correction

def _tpl_classic(v1, v2, fmt, p):
    x = f"({v1}-({fmt(p['Vpk'])}))"
    psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3)"
    gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
    # Fallback to 'lamb' if 'lambda_' isn't passed under that exact key
    lamb_val = p.get('lambda_', p.get('lamb', 0.0))
    drain = f"(tanh({fmt(p['alpha'])}*{v2}) * (1.0 + {fmt(lamb_val)}*{v2}))"
    return f"({gate} * {drain})"

def _tpl_modern(v1, v2, fmt, p):
    x = f"({v1}-({fmt(p['Vpk'])}))"
    psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3 + {fmt(p['P4'])}*{x}**4)"
    gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
    alpha_eff = f"({fmt(p['alpha'])} + {fmt(p['alpha_s'])} * (1.0 + tanh({psi})))"
    lamb_val = p.get('lambda_', p.get('lamb', 0.0))
    drain = f"(tanh({alpha_eff}*{v2}) * (1.0 + {fmt(lamb_val)}*{v2}))"
    return f"({gate} * {drain})"

def _tpl_6_term(v1, v2, fmt, p):
    x = f"({v1}-({fmt(p['Vpk'])}))"
    psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3 + {fmt(p['P4'])}*{x}**4 + {fmt(p['P5'])}*{x}**5 + {fmt(p['P6'])}*{x}**6)"
    gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
    return f"({gate} * tanh({fmt(p['alpha'])}*{v2}) * (1.0 + {fmt(p.get('lambda_', 0.0))}*{v2}))"

def _tpl_9_term(v1, v2, fmt, p):
    x = f"({v1}-({fmt(p['Vpk'])}))"
    psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3 + {fmt(p['P4'])}*{x}**4 + {fmt(p['P5'])}*{x}**5 + {fmt(p['P6'])}*{x}**6 + {fmt(p['P7'])}*{x}**7 + {fmt(p['P8'])}*{x}**8 + {fmt(p['P9'])}*{x}**9)"
    gate = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})))"
    return f"({gate} * tanh({fmt(p['alpha'])}*{v2}) * (1.0 + {fmt(p.get('lambda_', 0.0))}*{v2}))"

def _tpl_dual_peak(v1, v2, fmt, p):
    x1 = f"({v1}-({fmt(p['Vpk1'])}))"
    psi1 = f"({fmt(p['P1_1'])}*{x1} + {fmt(p['P2_1'])}*{x1}**2 + {fmt(p['P3_1'])}*{x1}**3)"
    ich1 = f"({fmt(p['Ipk1'])} * (1.0 + tanh({psi1})))"
    x2 = f"({v1}-({fmt(p['Vpk2'])}))"
    psi2 = f"({fmt(p['P1_2'])}*{x2})"
    ich2 = f"({fmt(p['Ipk2'])} * (1.0 + tanh({psi2})))"
    return f"(({ich1} + {ich2}) * tanh({fmt(p['alpha'])}*{v2}) * (1.0 + {fmt(p.get('lambda_', 0.0))}*{v2}))"

def _tpl_correction(v1, v2, fmt, p):
    x = f"({v1}-({fmt(p['Vpk'])}))"
    psi = f"({fmt(p['P1'])}*{x} + {fmt(p['P2'])}*{x}**2 + {fmt(p['P3'])}*{x}**3)"
    base_ids = f"({fmt(p['Ipk'])} * (1.0 + tanh({psi})) * tanh({fmt(p['alpha'])}*{v2}) * (1.0 + {fmt(p.get('lambda_', 0.0))}*{v2}))"
    # Softplus equivalent for equation parsers
    correction = f"({fmt(p['C_alpha'])} * ln(1.0 + exp({fmt(p['C_beta'])} * ({v1} - ({fmt(p['C_vshift'])})))))"
    return f"({base_ids} + {correction})"

def _tpl_mod1_angelov(v1, v2, fmt, p):
    # v1 = Vgs, v2 = Vds
    Vpk = f"({fmt(p['Vpks'])} - {fmt(p['deltVpks'])} + {fmt(p['deltVpks'])} * tanh({fmt(p['alphaS'])} * {v2}))"
    Vgsp = f"({v1} - {Vpk})"
    
    safe_n = f"({fmt(p['n'])} + 1e-12)"
    x = f"({fmt(p['n'])} * {Vgsp})"
    
    # torch.logaddexp(x, -x) translates to ln(exp(x) + exp(-x))
    Vgspa = f"((1.0 / {safe_n}) * (ln(exp({x}) + exp(-{x})) - ln(2.0)))"
    
    Veffp1 = f"(0.5 * ({Vgsp} - {Vgspa}))"
    Veffp2 = f"(0.5 * ({fmt(p['Vgsf'])} + {Vgspa}))"
    
    common_term = f"(1.0 + tanh({fmt(p['alphaS'])} * {v2}))"
    P1m = f"({fmt(p['P1'])} * (1.0 + {fmt(p['deltP1'])}) * {common_term})"
    P21m = f"({fmt(p['P21'])} * (1.0 + {fmt(p['deltP21'])}) * {common_term})"
    P22m = f"({fmt(p['P22'])} * (1.0 + {fmt(p['deltP22'])}) * {common_term})"
    P31m = f"({fmt(p['P31'])} * (1.0 + {fmt(p['deltP31'])}) * {common_term})"
    P32m = f"({fmt(p['P32'])} * (1.0 + {fmt(p['deltP32'])}) * {common_term})"
    
    P111 = f"({fmt(p['P1'])} * {fmt(p['Ipk'])} / ({fmt(p['Ipk1'])} + 1e-12))"
    
    ph1 = f"({P1m} * {Veffp1} + {P21m} * {Veffp1}**2 + {P31m} * {Veffp1}**3)"
    ph2 = f"({P111} * {Veffp2} + {P22m} * {Veffp2}**2 + {P32m} * {Veffp2}**3)"
    
    Ids1 = f"({fmt(p['Ipk'])} * (1.0 + tanh({ph1})) + {fmt(p['Ipk1'])} * tanh({ph2}))"
    alpha_eff = f"({fmt(p['alphaR'])} + {fmt(p['alphaS'])} * (1.0 + tanh({ph1})))"
    Ids2 = f"(tanh({alpha_eff} * {v2}) * (1.0 + {fmt(p['lambda_'])} * {v2}))"
    
    return f"({Ids1} * {Ids2})"

def _tpl_mod2_angelov(v1, v2, fmt, p):
    # v1 = Vgs, v2 = Vds
    Vgsp1 = f"({v1} - {fmt(p['Vpk1'])})"
    Vgsp2 = f"({v1} - {fmt(p['Vpk2'])})"
    Vgsp3 = f"({v1} - {fmt(p['Vpk3'])})"
    
    Pk1 = f"({fmt(p['K10'])} + ({fmt(p['K10'])} + {fmt(p['K11'])} * {v2}) * tanh({fmt(p['alpha1'])} * {v2}))"
    Pk2 = f"({fmt(p['K20'])} + ({fmt(p['K20'])} + {fmt(p['K21'])} * {v2}) * tanh({fmt(p['alpha2'])} * {v2}))"
    Pk3 = f"({fmt(p['K30'])} + ({fmt(p['K30'])} + {fmt(p['K31'])} * {v2}) * tanh({fmt(p['alpha3'])} * {v2}))"
    
    PhiP = f"({Pk1} * {Vgsp1} + {Pk2} * ({Vgsp2})**2 + {Pk3} * ({Vgsp3})**3)"
    Mipk = f"(1.0 + 0.5 * (1.0 + {fmt(p['Mipk1'])}) * tanh({fmt(p['qm'])} * ({v1} - {fmt(p['Vgsm'])})))"
    
    Ids1 = f"({fmt(p['Ipk'])} * (1.0 + {Mipk} * tanh({PhiP})))"
    Ids2 = f"(tanh({fmt(p['alpha'])} * {v2}))"
    
    return f"({Ids1} * {Ids2})"

def _tpl_mod3_angelov(v1, v2, fmt, p):
    # v1 = Vgs, v2 = Vds
    Ipk = f"({fmt(p['I0'])} + {fmt(p['I1'])} * {v2} + {fmt(p['I2'])} * {v2}**2 + {fmt(p['I3'])} * {v2}**3)"
    Zm = f"(({fmt(p['Pz0'])} + {fmt(p['Pz1'])} * {v2}) * tanh({fmt(p['alphaZ'])} * {v2}) + {fmt(p['Pz0'])})"
    PhiM = f"({Zm} * ({v1} - {fmt(p['Vgm0'])}))"
    
    Mpk = f"({fmt(p['Mpk0'])} + {fmt(p['MpkA'])} * tanh({PhiM}))"
    Vgsp = f"({v1} - {fmt(p['Vpk'])})"
    PhiP = f"({fmt(p['P1m'])} * {Vgsp} + {fmt(p['P2'])} * ({Vgsp})**2 + {fmt(p['P3'])} * ({Vgsp})**3)"
    
    Ids1 = f"({Ipk} * (1.0 + {Mpk} * tanh({PhiP})))"
    
    # Note: Removed PyTorch's `clamp` here. If your parser supports max/min, 
    # you can wrap this as f"max(-20.0, min(20.0, {fmt(p['alpha'])} * {v2}))"
    alpha_vds = f"({fmt(p['alpha'])} * {v2})"
    Ids2 = f"(tanh(sinh({alpha_vds})))"
    
    return f"({Ids1} * {Ids2})"



def classic_curtice(X, Beta, lambda_, alphaR, alphaS, P0, deltP0, P1, deltP1, P2, deltP2, P3, deltP3):
    Vgs, Vds = X[:, 0], X[:, 1]
    
    tanh_vds = torch.tanh(Vds)
    
    alpha = alphaR + alphaS * (1.0 + tanh_vds)
    A0 = P0 + deltP0 * (1.0 + tanh_vds)
    A1 = P1 + deltP1 * (1.0 + tanh_vds)
    A2 = P2 + deltP2 * (1.0 + tanh_vds)
    A3 = P3 + deltP3 * (1.0 + tanh_vds)
    
    # Using corrected polynomial: A1*Vgs + A2*Vgs^2 
    Ids1 = Beta * (A0 + A1 * Vgs + A2 * Vgs**2 + A3 * Vgs**3)
    Ids2 = torch.tanh(alpha * Vds) * (1.0 + lambda_ * Vds)
    
    return Ids1 * Ids2

def _tpl_classic_curtice(v1, v2, fmt, p):
    tanh_vds = f"tanh({v2})"
    
    alpha = f"({fmt(p['alphaR'])} + {fmt(p['alphaS'])}*(1.0 + {tanh_vds}))"
    A0 = f"({fmt(p['P0'])} + {fmt(p['deltP0'])}*(1.0 + {tanh_vds}))"
    A1 = f"({fmt(p['P1'])} + {fmt(p['deltP1'])}*(1.0 + {tanh_vds}))"
    A2 = f"({fmt(p['P2'])} + {fmt(p['deltP2'])}*(1.0 + {tanh_vds}))"
    A3 = f"({fmt(p['P3'])} + {fmt(p['deltP3'])}*(1.0 + {tanh_vds}))"
    
    Ids1 = f"({fmt(p['Beta'])} * ({A0} + {A1}*{v1} + {A2}*{v1}**2 + {A3}*{v1}**3))"
    lamb_val = p.get('lambda_', p.get('lamb', 0.0))
    Ids2 = f"(tanh({alpha}*{v2}) * (1.0 + {fmt(lamb_val)}*{v2}))"
    
    return f"({Ids1} * {Ids2})"

# ==========================================
# MODIFIED CURTICE FUNCTIONS
# ==========================================
def modified_curtice(X, Beta, Vst, Delta, lambda_, Vt0, gama1, Vk0, gama2, alphaR, alphaS):
    Vgs, Vds = X[:, 0], X[:, 1]
    
    Vt = Vt0 + gama1 * Vds
    Vk = Vk0 + gama2 * Vds
    alpha = alphaR + alphaS * (1.0 + torch.tanh(Vgs))
    
    Vgs1 = Vgs - Vt
    
    sqrt1 = torch.sqrt((Vgs1 - Vk)**2 + Delta**2)
    sqrt2 = torch.sqrt(Vk**2 + Delta**2)
    Vgs2 = Vgs1 - 0.5 * (Vgs1 + sqrt1 - sqrt2)
    
    # Clamp to prevent overflow in exp
    Vgs3 = Vst * torch.log(1.0 + torch.exp(torch.clamp(Vgs2 / Vst, max=80.0)))
    
    # Add small epsilon to prevent division by zero in Ids2
    Vgs3_safe = torch.clamp(Vgs3, min=1e-6)
    
    Ids1 = (Beta * Vgs3**2) / (1.0 + Vgs3**2)
    Ids2 = (1.0 + lambda_ * Vds) * torch.tanh(alpha * Vds / Vgs3_safe)
    
    return Ids1 * Ids2

def _tpl_modified_curtice(v1, v2, fmt, p):
    Vt = f"({fmt(p['Vt0'])} + {fmt(p['gama1'])}*{v2})"
    Vk = f"({fmt(p['Vk0'])} + {fmt(p['gama2'])}*{v2})"
    alpha = f"({fmt(p['alphaR'])} + {fmt(p['alphaS'])}*(1.0 + tanh({v1})))"
    
    Vgs1 = f"({v1} - {Vt})"
    sqrt1 = f"sqrt(({Vgs1} - {Vk})**2 + {fmt(p['Delta'])}**2)"
    sqrt2 = f"sqrt({Vk}**2 + {fmt(p['Delta'])}**2)"
    Vgs2 = f"({Vgs1} - 0.5 * ({Vgs1} + {sqrt1} - {sqrt2}))"
    
    Vgs3 = f"({fmt(p['Vst'])} * log(1.0 + exp({Vgs2} / {fmt(p['Vst'])})))"
    
    Ids1 = f"(({fmt(p['Beta'])} * {Vgs3}**2) / (1.0 + {Vgs3}**2))"
    lamb_val = p.get('lambda_', p.get('lamb', 0.0))
    Ids2 = f"((1.0 + {fmt(lamb_val)}*{v2}) * tanh({alpha} * {v2} / ({Vgs3} + 1e-6)))"
    
    return f"({Ids1} * {Ids2})"


# =============================================================================
#  alpha_eff helper — used by noNN_knee gate when knee_use_alpha_eff=True
# =============================================================================

def compute_alpha_eff(eq_name, phys_vgs, phys_vds, rp):
    """Compute the effective alpha that governs the Vds saturation knee.

    For models with a single alpha (classic_angelov, angelov_9_term, etc.)
    this returns that static scalar.  For models with Vgs/Vds-dependent
    alpha_eff (mod1_angelov, modern_angelov) it recomputes the intermediate
    physics terms from the frozen real-space params.

    All tensors in rp must already be frozen (requires_grad=False).

    Returns a tensor broadcastable with phys_vds.
    """
    if eq_name == 'mod1_angelov':
        alphaS  = rp['alphaS']
        alphaR  = rp['alphaR']
        Vpk     = rp['Vpks'] - rp['deltVpks'] + rp['deltVpks'] * torch.tanh(alphaS * phys_vds)
        Vgsp    = phys_vgs - Vpk
        n       = rp['n']
        x       = n * Vgsp
        Vgspa   = (1.0 / n) * (torch.logaddexp(x, -x) - math.log(2.0))
        Veffp1  = 0.5 * (Vgsp - Vgspa)
        tanh_aS_Vds = torch.tanh(alphaS * phys_vds)
        P1m     = rp['P1']  * (1 + rp['deltP1'])  * (1 + tanh_aS_Vds)
        P21m    = rp['P21'] * (1 + rp['deltP21']) * (1 + tanh_aS_Vds)
        P31m    = rp['P31'] * (1 + rp['deltP31']) * (1 + tanh_aS_Vds)
        ph1     = P1m * Veffp1 + P21m * Veffp1**2 + P31m * Veffp1**3
        return alphaR + alphaS * (1 + torch.tanh(ph1))

    elif eq_name == 'modern_angelov':
        Vpk     = rp['Vpk']
        dv      = phys_vgs - Vpk
        psi     = rp['P1']*dv + rp['P2']*dv**2 + rp['P3']*dv**3 + rp['P4']*dv**4
        return rp['alpha'] + rp['alpha_s'] * (1 + torch.tanh(psi))

    else:
        # Static fallback: single alpha parameter
        return rp.get('alpha', rp.get('alphaR', None))


def compute_vgs_gate(eq_name, phys_vgs, phys_vds, rp):
    """Compute h(Vgs, Vds) = 1 + tanh(psi) — the Vgs-dependent conduction gate.

    Mirrors the Ids1 factor in each Angelov-family model so the NN correction
    is naturally suppressed at pinch-off and amplified at peak gm.

    Returns a tensor broadcastable with phys_vgs, or ones if unsupported.
    """
    if eq_name == 'mod1_angelov':
        # psi = ph1 (depends on both Vgs and Vds through P1m, P21m, P31m)
        alphaS      = rp['alphaS']
        Vpk         = rp['Vpks'] - rp['deltVpks'] + rp['deltVpks'] * torch.tanh(alphaS * phys_vds)
        Vgsp        = phys_vgs - Vpk
        n           = rp['n']
        x           = n * Vgsp
        Vgspa       = (1.0 / n) * (torch.logaddexp(x, -x) - math.log(2.0))
        Veffp1      = 0.5 * (Vgsp - Vgspa)
        tanh_aS_Vds = torch.tanh(alphaS * phys_vds)
        P1m         = rp['P1']  * (1 + rp['deltP1'])  * (1 + tanh_aS_Vds)
        P21m        = rp['P21'] * (1 + rp['deltP21']) * (1 + tanh_aS_Vds)
        P31m        = rp['P31'] * (1 + rp['deltP31']) * (1 + tanh_aS_Vds)
        ph1         = P1m * Veffp1 + P21m * Veffp1**2 + P31m * Veffp1**3
        return 1.0 + torch.tanh(ph1)

    elif eq_name in ('classic_angelov', 'angelov_6_term', 'angelov_9_term'):
        dv  = phys_vgs - rp['Vpk']
        psi = rp['P1']*dv + rp['P2']*dv**2 + rp['P3']*dv**3
        return 1.0 + torch.tanh(psi)

    elif eq_name == 'modern_angelov':
        dv  = phys_vgs - rp['Vpk']
        psi = rp['P1']*dv + rp['P2']*dv**2 + rp['P3']*dv**3 + rp['P4']*dv**4
        return 1.0 + torch.tanh(psi)

    else:
        # Unsupported model — no Vgs gating (gate = 1 everywhere)
        return torch.ones_like(phys_vgs)


NONN_MODELS_CONFIG = {
    1: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
                'Vpk': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
                'alpha': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
                'P1': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'P2': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'P3': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'lambda_': {"min": -1.0, "max": 1.0, "lr": [0.01*10]} # Example of a static, un-trainable parameter!
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
                'Vpk': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
                'P1': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'P2': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'P3': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'P4': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'alpha_s': {"min": -10.0, "max": 10.0, "lr": [0.1*10]},
                'alpha': {"min": -10.0,  "max": 10.0, "lr": [0.1*10]},
            }
        }
    },
    2: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk': {"min": -10.0,  "max": 10.0, "lr": [0.01*10]},
                'Vpk': {"min": -10.0,  "max": 10.0, "lr": [0.01*10]},
                'alpha': {"min": -10.0,  "max": 10.0, "lr": [0.01*10]},
                'P1': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'P2': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'P3': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'lambda_': {"min": -1.0, "max": 1.0, "lr": [0.001*10]} # Example of a static, un-trainable parameter!
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk': {"min": -10.0,  "max": 10.0, "lr": [0.01*10]},
                'Vpk': {"min": -10.0,  "max": 10.0, "lr": [0.01*10]},
                'P1': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'P2': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'P3': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'P4': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'alpha_s': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
                'alpha': {"min": -10.0, "max": 10.0, "lr": [0.01*10]},
            }
        }
    },
    3: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk': {"min": -20.0,  "max": 20.0, "lr": [0.1*10]},
                'Vpk': {"min": -20.0,  "max": 20.0, "lr": [0.1*10]},
                'alpha': {"min": -20.0,  "max": 20.0, "lr": [0.1*10]},
                'P1': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'P2': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'P3': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'lambda_': {"min": -2.0, "max": 2.0, "lr": [0.01*10]} # Example of a static, un-trainable parameter!
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk': {"min": -20.0,  "max": 20.0, "lr": [0.1*10]},
                'Vpk': {"min": -20.0,  "max": 20.0, "lr": [0.1*10]},
                'P1': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'P2': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'P3': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'P4': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'alpha_s': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
                'alpha': {"min": -20.0, "max": 20.0, "lr": [0.1*10]},
            }
        }
    },
    4: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk': {"min": -20.0,  "max": 20.0, "lr": [0.01*10]},
                'Vpk': {"min": -20.0,  "max": 20.0, "lr": [0.01*10]},
                'alpha': {"min": -20.0,  "max": 20.0, "lr": [0.01*10]},
                'P1': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'P2': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'P3': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'lambda_': {"min": -2.0, "max": 2.0, "lr": [0.001*10]} # Example of a static, un-trainable parameter!
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk': {"min": -20.0,  "max": 20.0, "lr": [0.01*10]},
                'Vpk': {"min": -20.0,  "max": 20.0, "lr": [0.01*10]},
                'P1': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'P2': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'P3': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'P4': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'alpha_s': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
                'alpha': {"min": -20.0, "max": 20.0, "lr": [0.01*10]},
            }
        }
    },
    5: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk': {"min": -100.0,  "max": 100.0, "lr": [0.01*10]},
                'Vpk': {"min": -100.0,  "max": 100.0, "lr": [0.01*10]},
                'alpha': {"min": -100.0,  "max": 100.0, "lr": [0.01*10]},
                'P1': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'P2': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'P3': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'lambda_': {"min": -20.0, "max": 20.0, "lr": [0.001*10]} # Example of a static, un-trainable parameter!
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk': {"min": -100.0,  "max": 100.0, "lr": [0.01*10]},
                'Vpk': {"min": -100.0,  "max": 100.0, "lr": [0.01*10]},
                'P1': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'P2': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'P3': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'P4': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'alpha_s': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
                'alpha': {"min": -100.0, "max": 100.0, "lr": [0.01*10]},
            }
        }
    },
    6: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -8.0,  "max": 2.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.1,   "max": 5.0,  "lr": [0.01*10]},
                'P1':      {"min": 0.01,  "max": 3.0,  "lr": [0.01*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P3':      {"min": -0.5,  "max": 0.5,  "lr": [0.01*10]},
                'lambda_': {"min": -0.1,   "max": 0.1,  "lr": [0.001*10]}
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -8.0,  "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P2':      {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P3':      {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P4':      {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]}, # P4 shapes extreme tail, must be tiny
                'alpha_s': {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'lambda_': {"min": -0.1,   "max": 0.1,  "lr": [0.001*10]}
            }
        },
        'mod1_angelov': {
            'func': mod1_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'Ipk1':    {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'P1':      {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P21':     {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P22':     {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P31':     {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'P32':     {"min": -3.0,  "max": 3.0,  "lr": [0.01*10]},
                'Vpks':    {"min": -8.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltVpks':{"min": -2.0,  "max": 2.0,  "lr": [0.001*10]},
                'alphaR':  {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'alphaS':  {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'lambda_':   {"min": -0.1,  "max": 0.1,  "lr": [0.001*10]},
                'n':       {"min": 0.1,   "max": 5.0,  "lr": [0.01*10]},
                'Vgsf':    {"min": -8.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltP1':  {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'deltP21': {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'deltP22': {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'deltP31': {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'deltP32': {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
            }
        },
        'mod2_angelov': {
            'func': mod2_angelov,
            'bounds': {
                'Ipk':   {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'Mipk1': {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'qm':    {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'Vgsm':  {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'Vpk1':  {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'Vpk2':  {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'Vpk3':  {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'K10':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'K11':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'alpha1':{"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'K20':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'K21':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'alpha2':{"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'K30':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'K31':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'alpha3':{"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
            }
        },
        'mod3_angelov': {
            'func': mod3_angelov,
            'bounds': {
                'I0':    {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'I1':    {"min": -1.0, "max": 1.0, "lr": [0.01*10]},
                'I2':    {"min": -1.0, "max": 1.0, "lr": [0.01*10]},
                'I3':    {"min": -0.5, "max": 0.5, "lr": [0.01*10]},
                'Mpk0':  {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'MpkA':  {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'Vgm0':  {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'Vpk':   {"min": -8.0, "max": 2.0, "lr": [0.01*10]},
                'P1m':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'P2':    {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'P3':    {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'Pz0':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'Pz1':   {"min": -3.0, "max": 3.0, "lr": [0.01*10]},
                'alphaZ':{"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 5.0, "lr": [0.01*10]},
            }
        }
    },
    7: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -4.0,  "max": 0.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'P1':      {"min": 0.01,  "max": 3.0,  "lr": [0.01*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P3':      {"min": -0.5,  "max": 0.5,  "lr": [0.01*10]},
                'lambda_': {"min": -0.1,   "max": 0.1,  "lr": [0.001*10]}
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 5.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -8.0,  "max": 2.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.1,   "max": 5.0,  "lr": [0.01*10]},
                'P1':      {"min": 0.01,  "max": 3.0,  "lr": [0.01*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P3':      {"min": -0.5,  "max": 0.5,  "lr": [0.01*10]},
                'P4':      {"min": 0.0,  "max": 1.0,  "lr": [0.01*10]}, # P4 shapes extreme tail, must be tiny
                'lambda_': {"min": -0.1,   "max": 0.1,  "lr": [0.001*10]},
                'alpha_s': {"min": -4.0,   "max": 4.0,  "lr": [0.01*10]},
            }
        },
        'mod1_angelov': {
            'func': mod1_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'Ipk1':    {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P21':     {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P22':     {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P31':     {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'P32':     {"min": -1.0,  "max": 1.0,  "lr": [0.01*10]},
                'Vpks':    {"min": -5.0,  "max": 0.0,  "lr": [0.01*10]},
                'deltVpks':{"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'alphaR':  {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'alphaS':  {"min": 0.0,   "max": 3.0,  "lr": [0.01*10]},
                'lambda_':   {"min": -0.1,  "max": 0.1,  "lr": [0.001*10]},
                'n':       {"min": 0.5,   "max": 3.0,  "lr": [0.01*10]},
                'Vgsf':    {"min": -5.0,  "max": 0.0,  "lr": [0.01*10]},
                'deltP1':  {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'deltP21': {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'deltP22': {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'deltP31': {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'deltP32': {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
            }
        },
        'mod2_angelov': {
            'func': mod2_angelov,
            'bounds': {
                'Ipk':   {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'Mipk1': {"min": -1.0, "max": 1.0, "lr": [0.01*10]},
                'qm':    {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'Vgsm':  {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk1':  {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk2':  {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk3':  {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'K10':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'K11':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'alpha1':{"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'K20':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'K21':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'alpha2':{"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'K30':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'K31':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'alpha3':{"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
            }
        },
        'mod3_angelov': {
            'func': mod3_angelov,
            'bounds': {
                'I0':    {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'I1':    {"min": -0.5, "max": 0.5, "lr": [0.01*10]},
                'I2':    {"min": -0.5, "max": 0.5, "lr": [0.01*10]},
                'I3':    {"min": -0.2, "max": 0.2, "lr": [0.01*10]},
                'Mpk0':  {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'MpkA':  {"min": -1.0, "max": 1.0, "lr": [0.01*10]},
                'Vgm0':  {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk':   {"min": -5.0, "max": 0.0, "lr": [0.01*10]},
                'P1m':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'P2':    {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'P3':    {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'Pz0':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'Pz1':   {"min": -2.0, "max": 2.0, "lr": [0.01*10]},
                'alphaZ':{"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 3.0, "lr": [0.01*10]},
            }
        }
    },
    8: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -3.0,  "max": 0.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -3.0,  "max": 0.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P4':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]}, # P4 shapes extreme tail, must be tiny
                'alpha_s': {"min": 0.0,   "max": 2.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
        'mod1_angelov': {
            'func': mod1_angelov,
            'bounds': {
                'Ipk':     {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'Ipk1':    {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P21':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P22':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P31':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P32':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'Vpks':    {"min": -3.0,  "max": 0.0,  "lr": [0.01*10]},
                'deltVpks':{"min": -0.5,  "max": 0.5,  "lr": [0.0001*10]},
                'alphaR':  {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'alphaS':  {"min": 0.0,   "max": 2.0,  "lr": [0.01*10]},
                'lambda_':   {"min": -0.01, "max": 0.01, "lr": [0.0001*10]},
                'n':       {"min": 0.8,   "max": 2.0,  "lr": [0.001*10]},
                'Vgsf':    {"min": -3.0,  "max": 0.0,  "lr": [0.01*10]},
                'deltP1':  {"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
                'deltP21': {"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
                'deltP22': {"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
                'deltP31': {"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
                'deltP32': {"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
            }
        },
        'mod2_angelov': {
            'func': mod2_angelov,
            'bounds': {
                'Ipk':   {"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'Mipk1': {"min": -0.5, "max": 0.5, "lr": [0.001*10]},
                'qm':    {"min": 0.0,  "max": 2.0, "lr": [0.001*10]},
                'Vgsm':  {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk1':  {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk2':  {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk3':  {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'K10':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K11':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha1':{"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'K20':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K21':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha2':{"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'K30':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K31':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha3':{"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
            }
        },
        'mod3_angelov': {
            'func': mod3_angelov,
            'bounds': {
                'I0':    {"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'I1':    {"min": -0.2, "max": 0.2, "lr": [0.001*10]},
                'I2':    {"min": -0.2, "max": 0.2, "lr": [0.001*10]},
                'I3':    {"min": -0.1, "max": 0.1, "lr": [0.001*10]},
                'Mpk0':  {"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'MpkA':  {"min": -0.5, "max": 0.5, "lr": [0.001*10]},
                'Vgm0':  {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'Vpk':   {"min": -3.0, "max": 0.0, "lr": [0.01*10]},
                'P1m':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'P2':    {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'P3':    {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'Pz0':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'Pz1':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alphaZ':{"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
                'alpha': {"min": 0.0,  "max": 2.0, "lr": [0.01*10]},
            }
        }
    },
    9: {
        'classic_curtice': {
            'func': classic_curtice,
            'str_template': _tpl_classic_curtice,
            'bounds': {
                'Beta':    {"min": 0.01,  "max": 5.0,  "lr": [0.01*10]},
                'lambda_': {"min": -0.1,  "max": 0.1,  "lr": [0.001*10]},
                'alphaR':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]}, # Updated
                'alphaS':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]}, # Updated
                'P0':      {"min": -2.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltP0':  {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P1':      {"min": -2.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltP1':  {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -2.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltP2':  {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -2.0,  "max": 2.0,  "lr": [0.01*10]},
                'deltP3':  {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]}
            }
        },
        'modified_curtice': {
            'func': modified_curtice,
            'str_template': _tpl_modified_curtice,
            'bounds': {
                'Beta':    {"min": 0.01,  "max": 5.0,  "lr": [0.01*10]},
                'Vst':     {"min": 0.01,  "max": 1.0,  "lr": [0.001*10]},
                'Delta':   {"min": 0.01,  "max": 1.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.1,  "max": 0.1,  "lr": [0.001*10]},
                'Vt0':     {"min": -5.0,  "max": 0.0,  "lr": [0.01*10]},
                'gama1':   {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'Vk0':     {"min": -5.0,  "max": 0.0,  "lr": [0.01*10]},
                'gama2':   {"min": -0.5,  "max": 0.5,  "lr": [0.001*10]},
                'alphaR':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]}, # Updated
                'alphaS':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]}  # Updated
            }
        },
        'classic_angelov': {
            'func': classic_angelov,
            'str_template': _tpl_classic,
            'bounds': {
                'Ipk':     {"min": 0.5,   "max": 2.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -3.0,  "max": -1.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
    'angelov_6_term': {
        'func': angelov_6_term,
        'str_template': _tpl_6_term,
        'bounds': {
            'Ipk':     {"min": 0.5,   "max": 2.0,  "lr": [0.01*10]},
            'Vpk':     {"min": -3.0,  "max": -1.0, "lr": [0.01*10]},
            'alpha':   {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
            'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
            'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
            'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
            # Tighter bounds for higher order terms to prevent tanh saturation
            'P4':      {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
            'P5':      {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
            'P6':      {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
            'lambda_': {"min": -0.01, "max": 0.01, "lr": [0.0001*10]}
        }
    },
    'angelov_9_term': {
        'func': angelov_9_term,
        'str_template': _tpl_9_term,
        'bounds': {
            'Ipk':     {"min": 0.5,   "max": 2.0,   "lr": [0.01*10]},
            'Vpk':     {"min": -3.0,  "max": -1.0,  "lr": [0.01*10]},
            'alpha':   {"min": 0.2,   "max": 2.0,   "lr": [0.01*10]},
            'P1':      {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            'P2':      {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            'P3':      {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            'P4':      {"min": -0.1,  "max": 0.1,   "lr": [0.0001*10]},
            'P5':      {"min": -0.1,  "max": 0.1,   "lr": [0.0001*10]},
            'P6':      {"min": -0.1,  "max": 0.1,   "lr": [0.0001*10]},
            # Extreme restrictions on 7-9 to keep gradients alive
            'P7':      {"min": -0.01, "max": 0.01,  "lr": [0.00001*10]},
            'P8':      {"min": -0.01, "max": 0.01,  "lr": [0.00001*10]},
            'P9':      {"min": -0.01, "max": 0.01,  "lr": [0.00001*10]},
            'lambda_': {"min": -0.01, "max": 0.01,  "lr": [0.0001*10]}
        }
    },
    'angelov_dual_peak': {
        'func': angelov_dual_peak,
        'str_template': _tpl_dual_peak,
        'bounds': {
            'Ipk1':    {"min": 0.3,   "max": 1.5,   "lr": [0.01*10]},
            'Vpk1':    {"min": -3.0,  "max": -1.0,  "lr": [0.01*10]},
            'P1_1':    {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            'P2_1':    {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            'P3_1':    {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            
            # Second peak is usually smaller and occurs at a different voltage
            'Ipk2':    {"min": 0.01,  "max": 0.5,   "lr": [0.01*10]},
            'Vpk2':    {"min": -5.0,  "max": -2.0,  "lr": [0.01*10]},
            'P1_2':    {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
            
            'alpha':   {"min": 0.2,   "max": 2.0,   "lr": [0.01*10]},
            'lambda_': {"min": -0.01, "max": 0.01,  "lr": [0.0001*10]}
        }
    },
        'angelov_with_correction': {
                'func': angelov_with_correction,
                'str_template': _tpl_correction,
                'bounds': {
                    'Ipk':      {"min": 0.5,   "max": 2.0,   "lr": [0.01*10]},
                    'Vpk':      {"min": -3.0,  "max": -1.0,  "lr": [0.01*10]},
                    'alpha':    {"min": 0.2,   "max": 2.0,   "lr": [0.01*10]},
                    'P1':       {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
                    'P2':       {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
                    'P3':       {"min": -1.0,  "max": 1.0,   "lr": [0.001*10]},
                    'lambda_':  {"min": -0.01, "max": 0.01,  "lr": [0.0001*10]},
                    # --- New Correction Parameters ---
                    'C_alpha':  {"min": -0.5,  "max": 0.5,   "lr": [0.01*10]},
                    'C_beta':   {"min": 0.1,   "max": 10.0,  "lr": [0.05*10]},
                    'C_vshift': {"min": -3.0,  "max": 0.0,   "lr": [0.01*10]}
                }
            },
        'modern_angelov': {
            'func': modern_angelov,
            'str_template': _tpl_modern,
            'bounds': {
                'Ipk':     {"min": 0.5,   "max": 2.0,  "lr": [0.01*10]},
                'Vpk':     {"min": -3.0,  "max": -1.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P4':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]}, # P4 shapes extreme tail, must be tiny
                'alpha_s': {"min": 0.0,   "max": 2.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
        'mod1_angelov': {
            'func': mod1_angelov,
            'str_template': _tpl_mod1_angelov,
            'bounds': {
                'Ipk':     {"min": 0.5,   "max": 2.0,  "lr": [0.01*10]},
                'Ipk1':    {"min": 0.5,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P21':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P22':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P31':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P32':     {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'Vpks':    {"min": -3.0,  "max": -1.0, "lr": [0.01*10]},
                'deltVpks':{"min": -0.2,  "max": 0.2,  "lr": [0.0001*10]},
                'alphaR':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'alphaS':  {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'lambda_':   {"min": -0.01, "max": 0.01, "lr": [0.0001*10]},
                'n':       {"min": 1.0,   "max": 2.0,  "lr": [0.001*10]},
                'Vgsf':    {"min": -3.0,  "max": -1.0, "lr": [0.01*10]},
                'deltP1':  {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
                'deltP21': {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
                'deltP22': {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
                'deltP31': {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
                'deltP32': {"min": -0.1,  "max": 0.1,  "lr": [0.0001*10]},
            }
        },
        'mod2_angelov': {
            'func': mod2_angelov,
            'str_template': _tpl_mod2_angelov,
            'bounds': {
                'Ipk':   {"min": 0.5,  "max": 2.0, "lr": [0.01*10]},
                'Mipk1': {"min": -0.5, "max": 0.5, "lr": [0.001*10]},
                'qm':    {"min": 0.2,  "max": 2.0, "lr": [0.001*10]},
                'Vgsm':  {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'Vpk1':  {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'Vpk2':  {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'Vpk3':  {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'alpha': {"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
                'K10':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K11':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha1':{"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
                'K20':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K21':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha2':{"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
                'K30':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'K31':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alpha3':{"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
            }
        },
        'mod3_angelov': {
            'func': mod3_angelov,
            'str_template': _tpl_mod3_angelov,
            'bounds': {
                'I0':    {"min": 0.5,  "max": 2.0, "lr": [0.01*10]},
                'I1':    {"min": -0.2, "max": 0.2, "lr": [0.001*10]},
                'I2':    {"min": -0.2, "max": 0.2, "lr": [0.001*10]},
                'I3':    {"min": -0.1, "max": 0.1, "lr": [0.001*10]},
                'Mpk0':  {"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
                'MpkA':  {"min": -0.5, "max": 0.5, "lr": [0.001*10]},
                'Vgm0':  {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'Vpk':   {"min": -3.0, "max": -1.0,"lr": [0.01*10]},
                'P1m':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'P2':    {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'P3':    {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'Pz0':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'Pz1':   {"min": -1.0, "max": 1.0, "lr": [0.001*10]},
                'alphaZ':{"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
                'alpha': {"min": 0.2,  "max": 2.0, "lr": [0.01*10]},
            }
        }
    },
    10: {
        'classic_angelov': {
            'func': classic_angelov,
            'bounds': {
                'Ipk':     {"min": 0.5,   "max": 1.5,  "lr": [0.01*10]},
                'Vpk':     {"min": -2.0,  "max": -1.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
        'modern_angelov': {
            'func': modern_angelov,
            'bounds': {
                'Ipk':     {"min": 0.5,   "max": 1.5,  "lr": [0.01*10]},
                'Vpk':     {"min": -2.0,  "max": -1.0,  "lr": [0.01*10]},
                'alpha':   {"min": 0.2,   "max": 2.0,  "lr": [0.01*10]},
                'P1':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P2':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P3':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]},
                'P4':      {"min": -1.0,  "max": 1.0,  "lr": [0.001*10]}, # P4 shapes extreme tail, must be tiny
                'alpha_s': {"min": 0.0,   "max": 2.0,  "lr": [0.001*10]},
                'lambda_': {"min": -0.01,   "max": 0.01,  "lr": [0.0001*10]}
            }
        },
    }
}