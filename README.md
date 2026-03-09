# 💧 SHCO-Lite — System-of-Systems Hydrological-Computational Orchestration

> **Water-Aware Bit Migration for AI Data Centers**  
> A research framework linking urban hydrological monitoring with dynamic computational workload orchestration to combat the water consumption crisis in AI infrastructure.

---

## 🧠 What is SHCO-Lite?

Modern AI data centers consume billions of litres of water annually for cooling. SHCO-Lite is a **System-of-Systems** research framework that addresses this crisis by:

1. **Sensing** real-time water stress signals (clogs, leaks, floods) across data center cooling infrastructure
2. **Reasoning** about those signals using a secure, local RAG-based intelligence layer
3. **Acting** by migrating AI workloads geographically toward data centers in lower water-stress regions

The core innovation is moving beyond static, binary responses to a **Water-Aware Bit Migration** strategy — where AI tasks shift dynamically based on environmental stress detected via sparse sensor data.

---

## 📊 Research Context

| Metric | Value | Source |
|--------|-------|--------|
| Global DC water consumption (2023) | ~1.7 billion m³/year | Nature Communications 2024 |
| US DCs in water-stressed zones | ~1 in 5 | SEHN 2025 |
| Industry average WUE | 0.36 L/kWh | Lawrence Berkeley National Lab 2024 |
| Industry average PUE | 1.40 | Lawrence Berkeley National Lab 2024 |
| EU Climate Neutral DC Pact WUE target | 0.4 L/kWh | EU CNDCP 2024 |

**Modelled Data Centers** (Top 10 global markets by MW capacity, CBRE 2024):

| Data Center | Capacity (MW) | WUE (L/kWh) | Water Stress |
|-------------|--------------|-------------|--------------|
| DC_N_Virginia | 3,500 | 0.90 | 🟠 HIGH |
| DC_Beijing | 1,799 | 1.60 | 🟠 HIGH |
| DC_London | 1,600 | 0.45 | 🟢 LOW |
| DC_Singapore | 1,400 | 1.50 | 🟠 HIGH |
| DC_Tokyo | 1,350 | 0.55 | 🟢 LOW |
| DC_Frankfurt | 1,200 | 0.70 | 🟡 MEDIUM |
| DC_Shanghai | 1,100 | 1.40 | 🟠 HIGH |
| DC_Sydney | 900 | 1.20 | 🟠 HIGH |
| DC_Dallas | 850 | 1.80 | 🟠 HIGH |
| DC_Phoenix | 800 | 2.10 | 🔴 EXTREME |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SHCO-Lite Framework                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ simulation.py│    │  sensing.py  │    │intelligence  │  │
│  │              │───▶│              │───▶│    .py       │  │
│  │ Digital Twin │    │ Prototypical │    │  RAG + LLM   │  │
│  │ Monte Carlo  │    │  Networks    │    │  Lock-LLM    │  │
│  │  (Phase 1)   │    │  (Phase 2)   │    │  (Phase 3)   │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│                                    ┌─────────────▼──────┐  │
│                                    │  orchestration.py  │  │
│                                    │                    │  │
│                                    │   Bit Migration    │  │
│                                    │  Multi-Objective   │  │
│                                    │  Optimisation      │  │
│                                    │    (Phase 4)       │  │
│                                    └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Module Descriptions

| Module | Role | Key Technology |
|--------|------|----------------|
| `simulation.py` | Digital Twin — generates synthetic sensor data | Monte Carlo, NumPy |
| `sensing.py` | Perception Layer — detects anomalies with 5 examples | Prototypical Networks (Snell et al. 2017) |
| `intelligence.py` | Secure Brain — prescribes actions from emergency manual | RAG, Hugging Face, AES-256-GCM, SHA-256 |
| `orchestration.py` | Hands — migrates workloads to drier data centers | Multi-Objective Optimisation, Hysteresis |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 4060 or equivalent) for local LLM inference
- [Ollama](https://ollama.ai) or [LM Studio](https://lmstudio.ai) for local LLM backend

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/shco-lite.git
cd shco-lite

# Install the package and dependencies
pip install -e .

# Verify installation
python -c "import shcolite; print('SHCO-Lite ready.')"
```

---

## 🚀 Quick Start

### 1. Run the Digital Twin Simulation
```python
from shcolite.simulation import run_simulation

# Generates 20,160 rows of synthetic sensor data across 10 data centers
results = run_simulation(output_dir="shcolite_data", seed=42)

print(results["hydrological"].head())
print(results["computational"].head())
```

### 2. Run the Full Jupyter Demo
```bash
jupyter notebook notebooks/01_simulation_demo.ipynb
```

---

## 📁 Repository Structure

```
shco-lite/
│
├── README.md                        # This file
├── setup.py                         # Package installer
├── requirements.txt                 # All dependencies
├── .gitignore
│
├── shcolite/                        # Main Python package
│   ├── __init__.py
│   ├── simulation.py                ✅ Phase 1 — Digital Twin
│   ├── sensing.py                   🔄 Phase 2 — Anomaly Detection
│   ├── intelligence.py              🔄 Phase 3 — RAG Intelligence
│   └── orchestration.py             🔄 Phase 4 — Bit Migration
│
├── notebooks/
│   ├── 01_simulation_demo.ipynb
│   ├── 02_sensing_demo.ipynb
│   ├── 03_intelligence_demo.ipynb
│   └── 04_orchestration_demo.ipynb
│
├── docs/
│   ├── methodology.md
│   └── results.md
│
├── data/                            # Git-ignored (generated at runtime)
│   └── .gitkeep
│
└── tests/
    └── test_simulation.py
```

---

## 📈 Key Metrics

The primary impact metric of SHCO-Lite is **Litres of Water Saved per week** by migrating workloads away from high-stress data centers toward low-stress alternatives.

Baseline simulation results (vs. Phoenix worst-case WUE of 2.10 L/kWh):

```
DC_N_Virginia   [HIGH]    18,550 L/week  ███████████
DC_London       [LOW]     11,840 L/week  ███████
DC_Tokyo        [LOW]      9,360 L/week  █████
DC_Frankfurt    [MEDIUM]   7,516 L/week  ████
DC_Beijing      [HIGH]     3,814 L/week  ██
DC_Singapore    [HIGH]     3,608 L/week  ██
DC_Sydney       [HIGH]     3,560 L/week  ██
DC_Shanghai     [HIGH]     3,330 L/week  ██
DC_Dallas       [HIGH]     1,008 L/week  
DC_Phoenix      [HIGH]       161 L/week  

TOTAL POTENTIAL SAVINGS:  62,748 L/week
```

---

## 🔒 Security Architecture

The `intelligence.py` module implements a **Lock-LLM** security layer:

- **SHA-256 hashing** — verifies command integrity before execution
- **AES-256-GCM encryption** — protects sensitive sensor data in transit
- All LLM inference runs **fully locally** via Ollama/LM Studio — zero data leaves the machine

---

## 💻 Hardware Requirements

| Component | Minimum | Used In |
|-----------|---------|---------|
| CPU | 8 cores | Simulation, orchestration |
| RAM | 16 GB | All modules |
| GPU VRAM | 8 GB | LLM inference (intelligence.py) |
| Storage | 10 GB | Model weights + simulation data |

*Developed on: Intel i9 (24 cores) + NVIDIA RTX 4060 (8GB VRAM)*

---

## 📚 References

1. Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS.
2. Lawrence Berkeley National Laboratory (2024). *United States Data Center Energy Usage Report*.
3. CBRE Research (2024). *Global Data Center Trends 2024/2025*.
4. Li, P. et al. (2024). *Water consumption of data centers in China*. Nature Communications Earth & Environment.
5. SEHN (2025). *AI's Thirst for Water: Data Centers and the US Water Crisis*.
6. MSCI GeoSpatial Intelligence (2025). *Climate Risk and Data Center Infrastructure*.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

> **Academic Context:** VIT/NUS AI/ML Research Assignment | 15-min presentation + Jupyter Notebook demo  
> **Budget:** $0 — fully local, offline, open-source stack
