# QA formulas and gates

**Opposing-port imbalance**  
Δ_opp = max(|pN - pS|, |pE - pW|) / q̄

**Swirl index**  
W = sqrt((pN - pS)^2 + (pE - pW)^2) / (2·q̄)

**Default gates:** Δ_opp ≤ 0.01·q̄, W ≤ 0.002.  (See SOP §7, QA checklist.)  
Lag is evaluated for analysis overlays; **do not** apply any lag to the control signal in DCS.
