This is a comprehensive `README.md` file designed for your open-source repository. It is structured to cover both papers, clearly explaining the architecture (PNN), the optimization framework (NSGA-III), and the decision-making process (TOPSIS), while explicitly noting that the raw dataset is not included.

You can copy and paste the following content directly into your repository.

---

# PNN-NSGA-III Framework for Composite Stiffened Panel Optimization

This repository contains the official source code and implementation for the **Parallel Neural Network (PNN)** surrogate model and the **Multi-objective Optimization Framework (PNN-NSGA-III-TOPSIS)** for composite stiffened panels.

The code corresponds to the methods and algorithms described in the following research papers published in *Thin-Walled Structures*:

1. **[Paper 1]** *Parallel neural network feature extraction method for predicting buckling load of composite stiffened panels* (Vol. 199, 2024).
2. **[Paper 2]** *Multi-objective optimization of composite stiffened panels for mass and buckling load using PNN-NSGA-III algorithm and TOPSIS method* (Vol. 209, 2025).

---

## 📋 Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [Requirements](https://www.google.com/search?q=%23requirements)
* [Data Format](https://www.google.com/search?q=%23data-format)
* [Part 1: PNN Surrogate Model](https://www.google.com/search?q=%23part-1-pnn-surrogate-model)
* [Model Architecture](https://www.google.com/search?q=%23model-architecture)
* [Hyperparameter Tuning (Optuna)](https://www.google.com/search?q=%23hyperparameter-tuning-optuna)
* [Training](https://www.google.com/search?q=%23training)


* [Part 2: Multi-Objective Optimization](https://www.google.com/search?q=%23part-2-multi-objective-optimization)
* [NSGA-III Implementation](https://www.google.com/search?q=%23nsga-iii-implementation)
* [TOPSIS Decision Making](https://www.google.com/search?q=%23topsis-decision-making)


* [Citation](https://www.google.com/search?q=%23citation)
* [License](https://www.google.com/search?q=%23license)

---

## 🔍 Overview

This project addresses the design optimization of composite stiffened panels by combining deep learning with evolutionary algorithms.

1. **Prediction (PNN):** We utilize a Parallel Neural Network that processes heterogeneous data simultaneously:
* **T-Bi-LSTM:** Handles variable-length stacking sequences (using self-attention and residual normalization).
* **FNN:** Handles discrete variables (geometry, material properties).


2. **Optimization (NSGA-III):** We employ the Non-dominated Sorting Genetic Algorithm III to solve the conflict between maximizing **Buckling Load** and minimizing **Mass**.
3. **Decision Making (TOPSIS):** We use the Entropy Weight Method (EWM) and TOPSIS to objectively select the optimal design from the generated Pareto front.

**Note:** *The raw finite element analysis (FEA) datasets generated via ABAQUS are not included in this repository. Users must provide their own training data following the format specified below.*

---

## 📂 Project Structure

```text
.
├── data_loader/
│   └── dataset.py           # Custom Pytorch Dataset class for handling variable-length sequences
├── models/
│   ├── layers.py            # T-Bi-LSTM, Self-Attention, and Residual Block implementations
│   └── pnn.py               # Main Parallel Neural Network architecture
├── optimization/
│   ├── encoding.py          # Empty-ply encoding strategy for stacking sequences
│   ├── nsga3.py             # Implementation of NSGA-III (Reference point generation, Selection)
│   ├── operators.py         # SBX Crossover and Polymer Mutation (PLM) logic
│   └── constraints.py       # Stacking sequence design guidelines (symmetry, balance, etc.)
├── decision_making/
│   ├── entropy_weight.py    # Entropy Weight Method (EWM) calculation
│   └── topsis.py            # TOPSIS ranking implementation
├── tune_hyperparameters.py  # Optuna script for finding optimal PNN parameters
├── train_pnn.py             # Script to train the surrogate model
├── run_optimization.py      # Main script to run NSGA-III using trained PNN
└── requirements.txt         # Python dependencies

```

---

## 🛠 Requirements

The code is implemented in Python. Major dependencies include PyTorch for the neural network and Optuna for tuning.

```bash
pip install -r requirements.txt

```

*Example `requirements.txt` contents:*

```text
torch>=1.12.0
numpy
pandas
scipy
optuna
matplotlib
scikit-learn

```

---

## 📊 Data Format

Since the raw data is not provided, you must prepare your training data (CSV or JSON) with the following structure to use the `data_loader`:

| ID | Skin_Seq (String/List) | Stiffener_Seq (String/List) | Ply_Thickness (Float) | Mat_Prop_1 ... | Target_Buckling_Load |
| --- | --- | --- | --- | --- | --- |
| 1 | [45, -45, 0, 90...] | [0, 45, -45...] | 0.125 | 155.0 | 391.34 |

* **Stacking Sequences:** Should be lists of angles (e.g., `0, 45, -45, 90`).
* **Discrete Variables:** Ply thickness, material moduli (, etc.).
* **Target:** The Buckling Load (computed via FEM).

---

## 🧠 Part 1: PNN Surrogate Model

### Model Architecture

The PNN (Parallel Neural Network) is defined in `models/pnn.py`. It uses a **T-Bi-LSTM** (Bidirectional LSTM with Self-Attention) to extract features from stacking sequences and a separate Feed-Forward Network (FNN) for scalar inputs.

### Hyperparameter Tuning (Optuna)

We use **Optuna** to tune hyperparameters such as embedding size, hidden layer sizes, and learning rate.

To run the tuning process (referencing the procedure in Paper 1):

```bash
python tune_hyperparameters.py --n_trials 50 --study_name pnn_optimization

```

*This script optimizes for the lowest Validation Loss (MSE) and MAPE.*

### Training

Once hyperparameters are determined, train the final model:

```bash
python train_pnn.py --epochs 2000 --batch_size 64 --model_save_path saved_models/best_pnn.pth

```

*The model uses the SELU activation function and Adam optimizer with L2 regularization.*

---

## 🚀 Part 2: Multi-Objective Optimization

### NSGA-III Implementation

The optimization phase uses the trained PNN as a fast surrogate to evaluate fitness. The algorithm optimizes two objectives:

1. **Maximize** Buckling Load ()
2. **Minimize** Mass ()

To run the optimization:

```bash
python run_optimization.py --pop_size 500 --generations 50 --surrogate_path saved_models/best_pnn.pth

```

**Key Features Implemented:**

* **Encoding:** Handles variable length sequences using "empty ply" encoding (0=Empty, 1=0°, 2=45°, etc.).
* **Operators:** Simulated Binary Crossover (SBX) and Polynomial Mutation (PLM).
* **Constraints:** Penalty functions applied for violating engineering rules (Contiguity, 10% rule, Symmetry).

### TOPSIS Decision Making

After generating the Pareto front, the script automatically applies the decision-making logic:

1. **Entropy Weight Method:** Calculates objective weights based on data variance.
2. **TOPSIS:** Ranks the Pareto solutions by proximity to the ideal solution.

Results are saved to `optimization_results/best_designs.csv`.

---

## 📄 Citation

If you use this code or framework in your research, please cite the following papers:

```bibtex
@article{zhang2024parallel,
  title={Parallel neural network feature extraction method for predicting buckling load of composite stiffened panels},
  author={Zhang, Tao and Wang, Peiyan and Fu, Jianwei and Wang, Suian and Lian, Chenchen},
  journal={Thin-Walled Structures},
  volume={199},
  pages={111797},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.tws.2024.111797}
}

@article{zhang2025multi,
  title={Multi-objective optimization of composite stiffened panels for mass and buckling load using PNN-NSGA-III algorithm and TOPSIS method},
  author={Zhang, Tao and Wei, Zhao and Wang, Liping and Xue, Zhuo and Wang, Suian and Wang, Peiyan and Qi, Bowen and Yue, Zhufeng},
  journal={Thin-Walled Structures},
  volume={209},
  pages={112878},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.tws.2024.112878}
}

```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.