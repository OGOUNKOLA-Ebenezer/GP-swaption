#  Constrained Gaussian Process for Swaption Cube Calibration

###  A machine learning approach for building an arbitrage-free swaption cube under shape constraints

---

##  Overview

This project implements a **Constrained Gaussian Process (GP)** model for reconstructing a smooth, arbitrage-free **swaption cube** of prices (or implied volatilities).  
It enforces **financially consistent constraints** — monotonicity and convexity — directly during the optimization.

The model is based on a **finite-dimensional GP approximation** using a **Matérn 5/2 kernel** and **tri-linear hat basis functions**, following the methodology of *Bachoc et al. (2019)* and *Maatouk & Bay (2017)*.

---

##  Key Features

-  3D anisotropic Matérn 5/2 kernel (expiry × tenor × strike)  
- Linear constraints for **monotonicity** and **convexity** in strike  
-  Optional **in-plane consistency** across maturities  
-  MLE of hyperparameters and constrained MAP estimation  
-  Efficient and numerically stable optimization (with regularization)  
-  Visualization of reconstructed swaption surfaces  

---


