# People Trajectory Prediction Using a Self-Organizing Hybrid Approach

This repository contains the code and experimental material for the academic project:

<<<<<<< Updated upstream
Date limite : 18 janvier 2026
=======
**“People Trajectory Prediction Using a Self-Organizing Hybrid Approach”**  
Hybrid modeling combining **Boids simulation**, **LSTM neural networks**, and **Particle Swarm Optimization (PSO)**.
>>>>>>> Stashed changes

The project was carried out as part of the **Self-Organizing Systems (SOS)** course and simultaneously serves as an **individual research project** for the semester.

---

## 📌 Project Overview

Predicting the future trajectories of agents in a collective environment is a challenging task due to complex interactions and emergent behaviors.

This project proposes a **hybrid framework** composed of three main stages:

1. **Boids-based behavioral simulation** to generate realistic collective motion
2. **Particle Swarm Optimization (PSO)** to optimize:
   - Boids behavioral parameters
   - LSTM hyperparameters
3. **LSTM-based trajectory prediction** using supervised learning on simulated trajectories

The objective is to demonstrate that combining **self-organizing models** with **deep learning** leads to more accurate and stable trajectory predictions.

---

## 🧠 Methodology

### 1. Boids Simulation
- Agents follow classic Boids rules: separation, alignment, cohesion
- Generates realistic collective trajectories
- Serves as a synthetic dataset generator

### 2. PSO Optimization
PSO is applied at two levels:
- **Boids parameter optimization** (collision reduction, dispersion control, polarization maximization)
- **LSTM hyperparameter optimization** (learning rate, hidden dimension, number of layers)

### 3. LSTM Trajectory Prediction
- Sliding-window supervised learning
- Input: past positions of an agent
- Output: next predicted position
- Evaluation using MSE and ADE metrics

---

## 📂 Project Structure


│
├── data/
│ ├── boids_trajectories.npy
│ ├── boids_trajectories_best_pso.npy
│ ├── X.npy
│ └── Y.npy
│
├── models/
│ ├── best_boids_pso_params.npy
│ ├── best_lstm_pso_config*.npy
│ ├── lstm_clean_ .pth
│ └── lstm_optimized_ .pth
│
├── results/
│ ├── boids_baseline.gif
│ ├── boids_pso_optimized.gif
│ ├── curves_ .png
│ ├── loss_curve_ .png
│ └── metrics_ .npz
│
├── runs/
│ └── lstm_clean_ /
│
├── src/
│ ├── boid.py
│ ├── boids_metrics.py
│ ├── dataset.py
│ ├── simulation.py
│ ├── simulation_core.py
│ ├── pso_optimize_boids.py
│ ├── pso_lstm.py
│ ├── train_lstm_clean.py
│ ├── train_lstm_optimized.py
│ └── test_lstm.py
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore



---

## 🚀 How to Run the Project

### 1. Install Dependencies

pip install -r requirements.txt
