# Physics-Informed Neural Networks for Thermodynamic Property Prediction and Rankine Cycle Optimization

## Overview

This repository implements **phase-specific neural networks** to predict thermodynamic properties (molar volume) with Maxwell relations embedded in the loss function. The framework enables **real-time Rankine cycle simulation, multi-fluid screening, and cycle optimization**—achieving 100–250× computational speedup over traditional property databases like CoolProp.

## Key Features

✅ **Thermodynamically Consistent Models**
- Separate phase-specific networks (liquid and gas)
- Maxwell relations embedded as physics-informed loss penalties
- Internal consistency guaranteed across all derived properties

✅ **Fast Property Derivation**
- Predict molar volume \(V_m(T, P)\) via neural networks
- Derive enthalpy, entropy, and other properties using Maxwell relations and numerical integration
- Sub-millisecond property evaluations enabling real-time applications

✅ **High Accuracy**
- Liquid phase MAPE: <2%
- Gas phase MAPE: <4%
- Excellent generalization to unseen fluids (<2% on test set)

✅ **Practical Applications**
1. **Rankine Cycle Optimization** – Find optimal pressures via differential evolution (600× faster)
2. **ORC Fluid Screening** – Evaluate 28+ candidate fluids in seconds (5,000× faster)
3. **Model Interpretability** – SHAP analysis reveals feature importance and model transparency

## Repository Structure

```
├── train.ipynb                          # Neural network training pipeline
│   ├── Load and preprocess data
│   ├── Train liquid and gas models
│   ├── Embed Maxwell relations in loss function
│   └── Validate on held-out fluids
│
├── rankine_applications.ipynb           # Cycle applications
│   ├── rankine_cycle_simulation()    # Evaluate 4-state cycle
│   ├── rankine_optimization()        # Differential evolution optimization
│   └── orc_screening()               # Multi-fluid screening
│
├── shap_analysis.ipynb                  # Model explainability
│   ├── SHAP summary plots
│   ├── Feature importance ranking
│   ├── Dependence analysis
│   └── Interaction effects
│
└── media/
    ├── shap_analysis_outputs/        # SHAP visualizations
    │   ├── shap_summary_plot.jpeg
    │   ├── shap_bar_plot.jpeg
    │   ├── shap_dependence_*.jpeg
    │   └── shap_comparison_multiple_samples.jpeg
    │
    └── evaluation_screenshots/       # Model performance results
        ├── model_performance_metrics.png
        ├── cycle_optimization_convergence.png
        └── fluid_screening_results.png
```
## Pre-trained Models & Datasets

All trained models, scalers, and datasets are hosted on Google Drive for easy download and reproducibility.

### Download Links

| Asset | Description | Link |
|-------|-------------|------|
| **Liquid Phase Model** | Trained neural network for molar volume prediction (liquid) | [Download](https://drive.google.com/file/d/1mytiDz12ZQPJtimZuxK4nuQ45HK0MYvI/view?usp=sharing) |
| **Gas Phase Model** | Trained neural network for molar volume prediction (gas) | [Download](https://drive.google.com/file/d/15NLZpMZGntt7xY5DB2_Q_DmWsVMDcOlI/view?usp=sharing) |
| **Liquid Scaler** | StandardScaler for liquid phase input normalization | [Download](https://drive.google.com/file/d/1GqUSmTpmEV3NPv4aMXcAPRZASVxubGe0/view?usp=sharing) |
| **Gas Scaler** | StandardScaler for gas phase input normalization | [Download](https://drive.google.com/file/d/1hrKUvA_e-lucWgkUkDD_uvQVvO_TbgPc/view?usp=sharing) |
| **Gas Training Dataset** | Complete training data (28 fluids) | [Download](https://drive.google.com/file/d/1hh6DTs-zeRFwv7dGTqzf21MHG0eTCUHq/view?usp=sharing) |
| **Liquid Training Dataset** | Complete training data (28 fluids) | [Download](https://drive.google.com/file/d/18sYSdFxCP3K6xfBLxYuxmx95HyMWJDK4/view?usp=sharing) |


Or, you can access the complete folder here: [Download](https://drive.google.com/drive/folders/1d4DgUn1Q4SfVIn208R_clCX9pGSHt5ak?usp=sharing)
## Getting Started

### Installation

```bash
git clone https://github.com/ShlokP06/Rankine_Optimization.git
cd Rankine_Optimization
```

### Requirements
- Python 3.8+
- TensorFlow/PyTorch
- RDKit
- CoolProp (for validation)
- SHAP (for interpretability)
- SciPy (differential evolution, numerical integration)
- NumPy, Pandas, Matplotlib


#### 2. Rankine Cycle Optimization
```python
result = rankine_optimization(
    fluid_name='Pentane',
    T_evap_target=400,  # K
    T_cond_target=300,  # K
    maxiter=100
)
print(f"Optimal efficiency: {result['eta_opt']*100:.2f}%")
```

#### 3. Multi-Fluid ORC Screening
```python
top_fluids = orc_screening(
    candidate_fluids=['Pentane', 'Hexane', 'Toluene', 'R245fa'],
    P_evap=15e5,  
    P_cond=1e5,
    num_conditions=500
)
```

#### 4. SHAP Explainability
```python
shap_summary = analyze_model(
    model=liquid_model,
    X_test=test_data,
    plot_dir='media/shap_analysis_outputs'
)
```

## Model Performance

### Computational Performance

| Task | Traditional (CoolProp) | Neural Network | Speedup |
|------|----------------------|----------------|---------|
| Cycle Optimization (10,000 evals) | ~100 s | 20 ms | **5,000×** |
| ORC Screening (14,000 evals) | ~140 s | 0.028 s | **5,000×** |
| Per-evaluation time | ~10 ms | ~0.002 ms | **5,000×** |

## Physics-Informed Approach

The loss function embeds Maxwell relations to ensure thermodynamic consistency:

$$
\mathcal{L} = \text{MSE}(V_m) + \lambda \left\| \frac{\partial V_m}{\partial T} \right\|^2
$$


This ensures:
- Predictions satisfy fundamental thermodynamic laws
- No physically impossible states are predicted
- All derived properties (h, s, g) remain consistent

## Applications

### 1. Real-Time Rankine Cycle Optimization
- **Problem:** Find evaporator/condenser pressures maximizing thermal efficiency
- **Solution:** Differential evolution with neural net evaluations
- **Result:** Optimal operating conditions in <2 seconds (vs. >20 min with CoolProp)

### 2. Organic Rankine Cycle Screening
- **Problem:** Select best working fluid from 28+ candidates
- **Solution:** Batch evaluation with rapid property prediction
- **Result:** Complete screening in milliseconds; enables interactive fluid selection

### 3. Model Explainability
- **Problem:** Understand which fluid properties drive predictions
- **Solution:** SHAP analysis revealing feature importance
- **Result:** Temperature and critical volume are dominant; pressure plays minor role

## Limitations & Future Work

### Current Limitations
- **Fluid coverage:** 28 fluids; scaling to full industrial database ongoing
- **Supercritical region:** MAPE rises to ~2.4% near/above critical point
- **Transport properties:** Viscosity and thermal conductivity not yet modeled
- **Mixtures:** Single-component fluids only; multi-component extensions planned

### Future Directions
- Physics-informed neural networks (PINNs) with hard constraints
- Transport property models (viscosity, thermal conductivity)
- Binary and ternary mixture support
- Integration with ASPEN Plus and commercial process simulators

## License

MIT License – see LICENSE file for details

## Contact & Support

For questions, issues, or contributions:
- **Email:** parikh.shlokp@gmail.com
- **Documentation:** See `/docs` folder for detailed methodology

## Acknowledgments

Special thanks to Professor Dr. Gaurav Chauhan for guidance and support throughout this research.

---

**Transform Rankine cycle design from iterative guesswork to real-time, physics-informed optimization.**
