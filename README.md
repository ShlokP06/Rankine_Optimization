# Physics-Informed Neural Networks for Thermodynamic Property Prediction and Rankine Cycle Optimization

## Overview

Phase-specific neural networks predict thermodynamic properties (molar
volume) with Maxwell relations embedded directly in the loss function,
guaranteeing internally consistent derived properties. This enables
real-time Rankine cycle simulation, multi-fluid screening, and cycle
optimization at 100–5,000× the speed of traditional property databases
like CoolProp.

## Key Results

- **Thermodynamic consistency** — separate liquid/gas-phase networks, with
  Maxwell relations embedded as physics-informed loss penalties so enthalpy,
  entropy, and other derived properties stay internally consistent rather
  than being fit independently.
- **Accuracy** — liquid-phase MAPE < 2%, gas-phase MAPE < 4%, with < 2%
  generalization error on unseen fluids.
- **Speed** — sub-millisecond property evaluation once trained, vs. ~10 ms
  per CoolProp call.
- **Rankine cycle optimization** — differential-evolution search over
  evaporator/condenser pressures converges in <2 seconds using the neural
  surrogate, vs. >20 minutes with CoolProp (≈5,000× speedup on 10,000
  evaluations).
- **ORC fluid screening** — evaluates 28+ candidate organic Rankine cycle
  fluids in milliseconds instead of minutes, making interactive fluid
  selection practical.
- **Interpretability** — SHAP analysis on the property model shows
  temperature and critical volume dominate the prediction, with pressure
  playing a minor role.

## Physics-informed loss

The loss function embeds a Maxwell relation directly, so predictions are
constrained to physically consistent states rather than being penalized
for inconsistency only after the fact:

$$
\mathcal{L} = \text{MSE}(V_m) + \lambda \left\| \frac{\partial V_m}{\partial T} \right\|^2
$$

## Repository structure

```
Train.ipynb                  # Training pipeline: load data, train liquid/gas
                              # models, embed Maxwell relations in the loss,
                              # validate on held-out fluids
Rankine_Applications.ipynb   # rankine_cycle_simulation(), rankine_optimization()
                              # (differential evolution), orc_screening()
SHAP_Analysis.ipynb          # SHAP summary plots, feature importance,
                              # dependence and interaction analysis
Media output/
├── SHAP/                    # SHAP visualizations
└── evaluation screenshots/  # Performance metrics, convergence, screening plots
docs/
└── ML_Based_Rankine_Cycle_Optimization.pdf
```

## Pre-trained models & datasets

Trained models, scalers, and datasets are hosted on Google Drive:

| Asset | Description | Link |
|-------|-------------|------|
| Liquid Phase Model | Molar volume prediction (liquid) | [Download](https://drive.google.com/file/d/1mytiDz12ZQPJtimZuxK4nuQ45HK0MYvI/view?usp=sharing) |
| Gas Phase Model | Molar volume prediction (gas) | [Download](https://drive.google.com/file/d/15NLZpMZGntt7xY5DB2_Q_DmWsVMDcOlI/view?usp=sharing) |
| Liquid Scaler | Input normalization (liquid) | [Download](https://drive.google.com/file/d/1GqUSmTpmEV3NPv4aMXcAPRZASVxubGe0/view?usp=sharing) |
| Gas Scaler | Input normalization (gas) | [Download](https://drive.google.com/file/d/1hrKUvA_e-lucWgkUkDD_uvQVvO_TbgPc/view?usp=sharing) |
| Gas Training Dataset | 28 fluids | [Download](https://drive.google.com/file/d/1hh6DTs-zeRFwv7dGTqzf21MHG0eTCUHq/view?usp=sharing) |
| Liquid Training Dataset | 28 fluids | [Download](https://drive.google.com/file/d/18sYSdFxCP3K6xfBLxYuxmx95HyMWJDK4/view?usp=sharing) |

Or grab the [complete folder](https://drive.google.com/drive/folders/1d4DgUn1Q4SfVIn208R_clCX9pGSHt5ak?usp=sharing).

## Getting started

```bash
git clone https://github.com/ShlokP06/Rankine_Optimization.git
cd Rankine_Optimization
```

Requirements: Python 3.8+, TensorFlow/PyTorch, RDKit, CoolProp (for
validation), SHAP, SciPy (differential evolution, numerical integration),
NumPy, Pandas, Matplotlib.

```python
# Rankine cycle optimization
result = rankine_optimization(
    fluid_name='Pentane',
    T_evap_target=400,  # K
    T_cond_target=300,  # K
    maxiter=100
)
print(f"Optimal efficiency: {result['eta_opt']*100:.2f}%")

# Multi-fluid ORC screening
top_fluids = orc_screening(
    candidate_fluids=['Pentane', 'Hexane', 'Toluene', 'R245fa'],
    P_evap=15e5,
    P_cond=1e5,
    num_conditions=500
)

# SHAP explainability
shap_summary = analyze_model(
    model=liquid_model,
    X_test=test_data,
    plot_dir='Media output/SHAP'
)
```

## Limitations & future work

- Trained on 28 fluids; scaling to a full industrial fluid database is
  ongoing.
- MAPE rises to ~2.4% in the supercritical region, near/above the critical
  point.
- Transport properties (viscosity, thermal conductivity) aren't modeled yet.
- Single-component fluids only — binary/ternary mixture support is planned.
- Physics-informed hard constraints (vs. the current soft loss penalty) and
  integration with process simulators like ASPEN Plus are natural next
  steps.

## License

MIT — see `LICENSE`.

## Acknowledgments

Guided by Prof. Gaurav Chauhan.
