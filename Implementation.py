# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 08:36:16 2025

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import ArrayLike
from gp_swaption_cube import ConstrainedSwaptionGPCube, build_Phi
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
#from Usefuls import construct_full_dataset

# Example: Build a small cube and fit MAP on fake data
T = np.array([5, 10, 15, 20], dtype=float)
t = np.array([1, 5, 10, 20, 30], dtype=float)
K = np.linspace(-0.02, 0.02, 7)

# Create a toy dataset: sample a few points and pretend they're noisy prices
rng = np.random.RandomState(42)
n_obs = 40
X_obs = np.column_stack([
     rng.choice(T, size=n_obs),
     rng.choice(t, size=n_obs),
     rng.choice(K, size=n_obs),
   ])
# Toy "true" function: decreasing in K (payer-like), some convexity
def toy_price(x):
   T_, t_, K_ = x
   base = 0.02 * np.exp(-0.02 * (T_ + t_))
   smile = np.maximum(0.0, 0.02 - 5.0 * (K_ + 0.005) ** 2)
   return base + smile

y_obs = np.array([toy_price(x) for x in X_obs]) + 0.0005 * rng.randn(n_obs)

model = ConstrainedSwaptionGPCube(T, t, K, theta_T=0.4, theta_t=0.4, theta_K=0.4, sigma=0.5, noise=0.003, kind="payer")
# Optional MLE (on toy data this is not very meaningful)
model.fit_mle(y_obs, X_obs, restarts=3, seed=0)

# Compute MAP under constraints
model.fit_map(y_obs, X_obs, year_step_inplane=True)

# Predict on a few points
Xq = np.array([
        [10, 5, -0.01],
       [10, 5,  0.00],
       [10, 5,  0.01],
  ], dtype=float)
preds = model.predict(X_obs[3:6,:])
print("Predictions at query points:", preds)






# ============================================================
#  Load and prepare data
# ============================================================

df = pd.read_csv(r"C:\Users\Dell\Documents\ML_Project\GP_Regression\Swaption_price_simulated.txt")
df = df[:-1]  # Remove last row if needed

cols = ['Expiry', 'Tenor', 'Fwd', -150, -100, -50, -25, 0, 25, 50, 100, 150]
df.columns = cols

expiries = df['Expiry']
tenors = df['Tenor'].values
spreads = np.array([-150, -100, -50, -25, 0, 25, 50, 100, 150])
F = df['Fwd']

# Construct strikes matrix
n_obs = len(F)
n_spreads = len(spreads)
K = pd.DataFrame(index = F, columns = spreads) 

for spread in spreads:
    for f in F:
        K.loc[f, spread] = f + 0.0001*spread
        


# Get prices matrix
prices = df.loc[:, spreads]





# ============================================================
# Plot the prices  curves
# ============================================================

fig, ax = plt.subplots(7,5, figsize = (20,18),gridspec_kw={'hspace':0.5, 'wspace':0.5})
ax = ax.flatten()
for i in range(len(F)):
    ax[i].set_title( f"Expiry : {df.Expiry[i]} Tenor : {df.Tenor[i]} " )
    ax[i].scatter(K.iloc[i,:], prices.iloc[i,:])
    ax[i].plot(K.iloc[i,:], prices.iloc[i,:])
    ax[i].scatter(K.iloc[i,len(spreads)//2], prices.iloc[i,len(spreads)//2], c = 'red', label = ("ATM price"))
    ax[i].legend()
plt.show()    

# Function to check the monotoncity constraints of the prices for each (T,t)
def check_monotonicity(prices : ArrayLike, option_type = "payer")-> bool:
    prices = np.asarray(prices)
    if prices.ndim != 1:
        raise ValueError("The prices should have one dimension !")
    if option_type != "payer" and option_type != "receiver":
        raise ValueError(
            f"Optiontype  no valid : {option_type}"
            f"Choose between `payer` or `receiver`"
            )
    if option_type == "payer":
        return all(prices[i] > prices[i+1] for i in range(len(prices)-2))
    if option_type =="receiver":
        return all(prices[i] < prices[i+1] for i in range(len(prices)-2))

# Function to check the convexity constraints of the prices for each (T,t)   
def check_convexity(prices : ArrayLike, strikes:ArrayLike)-> bool:
    
    prices = np.asarray(prices)
    strikes = np.asarray(strikes)
    if prices.ndim != 1 or strikes.ndim != 1 or len(prices) != len(strikes):
        raise ValueError("Prices and Strikes must have one dimenson or same lenght")
    if len(prices) != len(strikes):
        raise ValueError("Prices and strikes should have the same lenght")
    pentes = np.diff(prices) / np.diff(strikes)
    
    return all( pentes[i] < pentes[i+1] for i in range(len(pentes) -2))

# Check the montocity and the convexity  in whole prices data. And get the percentage of (T,t) verifying the constraints
monotocity_counter = 0 # Count the number of (T,t) satisfying the monotocity in the prices
convexity_counter = 0  # Count the number of (T,t) satisfying the convexity in the prices 
for (row_price, row_strike) in zip( prices.itertuples(index = False), K.itertuples(index = False)):
    Price = np.array(row_price[:])
    Strikes = np.array(row_strike)
    if check_monotonicity(Price):
        monotocity_counter += 1 
    if check_convexity(Price, Strikes):
        convexity_counter += 1
        
# Percentage of (T,t) pairs that verify the monotocity and convexity check         
Total_pair = len(df)
mono_percent = (monotocity_counter/Total_pair)*100
conv_percent = (convexity_counter/Total_pair )*100
print("\n\n------------------------------CHECKING CONSTRAINTS------------------------------\n")
print(f"The percentage of the (T,t) that satisfy the monotocity constraints in the prices is {mono_percent:.2f}%\n")
print(f"The percentage of the (T,t) that satisfy the convexity constraints in the prices is {conv_percent:.2f}%\n")
print("---------------------------------------------------------------------------------------")

# ============================================================
#  Construct training data in (T, t, K, price) format
# ============================================================

def construct_full_dataset(expiries, tenors, K, prices):
    """
    Construct full dataset with all observations.
    Returns: X (N x 3), y (N,) where X = [T, t, K]
    """
    X_list = []
    y_list = []
    
    for i in range(len(expiries)):
        T = expiries[i]
        t = tenors[i]
        for j in range(K.shape[1]):
            strike = K.iloc[i, j]
            price = prices.iloc[i, j]
            X_list.append([T, t, strike])
            y_list.append(price)
    
    return np.array(X_list), np.array(y_list)

X_full, y_full = construct_full_dataset(expiries, tenors, K, prices)
print(f"Full dataset: {X_full.shape[0]} observations")
print(f"Price range: [{y_full.min():.6f}, {y_full.max():.6f}]")


# ============================================================
# Create sampling strategy per requirements
# ============================================================

def create_training_subset(expiries, tenors, K, prices, spreads, p=0.5, seed=42):
    """
    Create training subset according to specifications:
    - Always include ATM (spread=0) for key pairs
    - Always include at least one payer OTM and one receiver OTM for key pairs
    - Key pairs: (5,5), (5,10), (10,5), (10,10)
    - Random p% of ATM swaptions
    - Random p% of OTM and ITM payer swaptions
    
    Args:
        expiries: array of expiry times
        tenors: array of tenor times
        K: strikes matrix (n_obs x n_spreads)
        prices: prices matrix (n_obs x n_spreads)
        spreads: array of spread values
        p: percentage of data to include (0 to 1)
        seed: random seed
    
    Returns:
        X_train: (N x 3) array of [T, t, K]
        y_train: (N,) array of prices
        indices: selected indices for tracking
    """
    rng = np.random.RandomState(seed)
    
    # Key pairs that must have complete coverage
    key_pairs = [(5.0, 5.0), (5.0, 10.0), (10.0, 5.0), (10.0, 10.0)]
    
    selected_indices = []
    
    # Ensure spreads is an array
    spreads_array = np.atleast_1d(spreads)
    
    # Find ATM index (spread = 0)
    atm_idx = np.where(spreads_array == 0)[0][0]
    
    # Payer OTM: strikes > forward, so positive spreads
    payer_otm_indices = np.where(spreads_array > 0)[0]
    # Receiver OTM: strikes < forward, so negative spreads  
    receiver_otm_indices = np.where(spreads_array < 0)[0]
    
    # Step 1: Ensure key pairs have required observations
    for exp, ten in key_pairs:
        # Find matching observation
        mask = (expiries == exp) & (tenors == ten)
        obs_idx = np.where(mask)[0]
        
        if len(obs_idx) > 0:
            obs_idx = obs_idx[0]
            
            # Always include ATM
            selected_indices.append((obs_idx, atm_idx))
            
            # Include at least one payer OTM (randomly choose one)
            payer_spread_idx = rng.choice(payer_otm_indices)
            selected_indices.append((obs_idx, payer_spread_idx))
            
            # Include at least one receiver OTM (randomly choose one)
            receiver_spread_idx = rng.choice(receiver_otm_indices)
            selected_indices.append((obs_idx, receiver_spread_idx))
    
    # Step 2: Random p% of ATM swaptions
    all_atm_indices = []
    for i in range(len(expiries)):
        all_atm_indices.append((i, atm_idx))
    
    # Remove already selected ATM from key pairs
    remaining_atm = [idx for idx in all_atm_indices if idx not in selected_indices]
    n_atm_to_add = int(p * len(remaining_atm))
    selected_atm_indices = rng.choice(len(remaining_atm), size=n_atm_to_add, replace=False)
    selected_indices.extend([remaining_atm[i] for i in selected_atm_indices])
    
    # Step 3: Random p% of payer OTM and ITM swaptions
    # Payer swaptions: all positive and negative spreads
    all_payer_indices = []
    for i in range(len(expiries)):
        for j in range(len(spreads)):
            if j != atm_idx:  # Exclude ATM already handled
                all_payer_indices.append((i, j))
    
    # Remove already selected
    remaining_payer = [idx for idx in all_payer_indices if idx not in selected_indices]
    n_payer_to_add = int(p * len(remaining_payer))
    selected_payer_indices = rng.choice(len(remaining_payer), size=n_payer_to_add, replace=False)
    selected_indices.extend([remaining_payer[i] for i in selected_payer_indices])
    
    # Convert selected indices to training data
    X_train = []
    y_train = []
    
    for obs_idx, spread_idx in selected_indices:
        T = expiries[obs_idx]
        t = tenors[obs_idx]
        strike = K.iloc[obs_idx, spread_idx]
        price = prices.iloc[obs_idx, spread_idx]
        
        X_train.append([T, t, strike])
        y_train.append(price)
    
    return np.array(X_train), np.array(y_train), selected_indices

# ============================================================
#  Define grid for basis functions
# ============================================================

def create_basis_grid(expiries, tenors, K, 
                     n_T=15, n_t=12, n_K=20):
    """
    Create regular grid for basis functions SN.
    
    Args:
        expiries, tenors, K: data arrays
        n_T: number of grid points in expiry dimension
        n_t: number of grid points in tenor dimension  
        n_K: number of grid points in strike dimension
    
    Returns:
        T_grid, t_grid, K_grid: 1D arrays defining the grid
    """
    # Define grid bounds with slight padding
    T_min, T_max = expiries.min(), expiries.max()
    t_min, t_max = tenors.min(), tenors.max()
    K_min, K_max = K.min().min(), K.max().max()
    
    # Create regular grids
    T_grid = np.linspace(T_min, T_max, n_T)
    t_grid = np.linspace(t_min, t_max, n_t)
    K_grid = np.linspace(K_min, K_max, n_K)
    
    print(f"Grid dimensions: T={n_T}, t={n_t}, K={n_K}")
    print(f"Total basis functions: {n_T * n_t * n_K}")
    print(f"Grid ranges:")
    print(f"  T: [{T_min:.3f}, {T_max:.3f}]")
    print(f"  t: [{t_min:.3f}, {t_max:.3f}]")
    print(f"  K: [{K_min:.4f}, {K_max:.4f}]")
    
    return T_grid, t_grid, K_grid

T_grid, t_grid, K_grid = create_basis_grid(expiries, tenors, K, 
                                           n_T=8, n_t=7, n_K=11)

# ============================================================
#  Train the model
# ============================================================


   


# Create training subset
print("\n" + "="*60)
print("Creating training subset")
print("="*60)

X_train, y_train, train_indices = create_training_subset(
    expiries, tenors, K, prices, spreads, p=0.5, seed=42
)

print(f"Training set size: {len(y_train)} observations")
print(f"Percentage of full data: {100*len(y_train)/len(y_full):.1f}%")

# Initialize model
model = ConstrainedSwaptionGPCube(
        T_grid=T_grid,
        t_grid=t_grid, 
        K_grid=K_grid,
        theta_T=0.3,
        theta_t=0.3,
        theta_K=0.3,
        sigma=1,
        noise=0.01,
        kind='payer'
    ) 
model.fit_mle(y_train, X_train)   
model.fit_map(y_train, X_train, year_step_inplane=False) 
#pred = model.predict(X_train)   

from scipy.optimize import linprog


# Assurez-vous d'avoir accès à l'objet LinearConstraint
# Ici, nous récupérons A et lb/ub du LinearConstraint précédemment construit
# Supposons que votre contrainte unique est stockée dans la variable `constraint_object`
cons = model.build_constraints(year_step_inplane = False)
constraint_object = cons[0] 

# Récupérer les composantes de la contrainte (A, lb, ub)
A_constr = constraint_object.A
lb_constr = constraint_object.lb
ub_constr = constraint_object.ub
N_variables = A_constr.shape[1]
N_contraintes = A_constr.shape[0]

# --- Étape 1 : Préparer pour linprog ---

# linprog requiert A_ub * xi <= b_ub (inégalités) et A_eq * xi = b_eq (égalités)

# 1. Fonction objectif factice (c = 0)
c = np.zeros(N_variables)

# 2. Conversion des contraintes A * xi >= lb en -A * xi <= -lb
is_lower_bound = ~np.isinf(lb_constr)
A_ub = -A_constr[is_lower_bound, :]
b_ub = -lb_constr[is_lower_bound]

# 3. Conversion des contraintes A * xi <= ub en A * xi <= ub (si présentes)
is_upper_bound = ~np.isinf(ub_constr)
A_ub_upper = A_constr[is_upper_bound, :]
b_ub_upper = ub_constr[is_upper_bound]

# Fusionner toutes les contraintes d'inégalité (A*xi <= b)
A_linprog = np.vstack([A_ub, A_ub_upper])
b_linprog = np.hstack([b_ub, b_ub_upper])

# Pour cet exemple, nous supposons qu'il n'y a pas de contraintes d'égalité, 
# car vos contraintes de monotonicité/convexité sont des inégalités pures.

# --- Étape 2 : Lancer le Solveur PL ---

print("Lancement du test de faisabilité avec linprog...")

res_pl = linprog(c, 
                 A_ub=A_linprog, 
                 b_ub=b_linprog, 
                 method='highs') # 'highs' est souvent le plus performant

# --- Étape 3 : Interpréter le Résultat ---

print("\n--- Résultat du Test de Faisabilité ---")
print(f"Statut : {res_pl.status}")
print(f"Message : {res_pl.message}")

if res_pl.status == 0:
    print("✅ CONSISTANCE CONFIRMÉE. La région réalisable n'est pas vide.")
elif res_pl.status == 2:
    print("❌ INCONSISTANCE CONFIRMÉE. linprog a trouvé la région infaisable.")
    print("   Le problème est mathématiquement impossible à résoudre.")
else:
    print(f"⚠️ Statut indéterminé ({res_pl.status}). Le solveur n'a pas pu conclure.")

# Si res_pl.status est 0, vous pouvez voir un point réalisable (res_pl.x)








#k = model.kernel 
#grid = model.grid
#P = grid.points()

#Matrix = k.K(P,P)

#TT, tt, KK = np.meshgrid(model.T_grid, model.t_grid, model.K_grid, indexing='ij')
#PP = np.column_stack((TT.ravel(), tt.ravel(), KK.ravel()))



#def is_positive_definite_cholesky(A):
#    # 1. Vérifier si la matrice est symétrique (une condition nécessaire)
#    if not np.allclose(A, A.T):
#        print("La matrice n'est pas symétrique.")
#        return False
#    
    # 2. Tenter la décomposition de Cholesky
#    try:
#        # La fonction cholesky renvoie l'erreur LinAlgError si A n'est pas DP
#        cholesky(A)
#        return True
#    except LinAlgError:
        # L'erreur signifie que le mineur principal est non positif
#        return False
#

#print(f"Matrice  est DP : {is_positive_definite_cholesky(Matrix)}")

#Phi = build_Phi(X_train, grid)

#A = Phi @ Matrix @ Phi.T + (0.002**2) * np.eye(len(y_train))

#print(f"A  est DP : {is_positive_definite_cholesky(A)}")

#from scipy.linalg import cho_factor, cho_solve, solve
#cF = cho_factor(A, lower=True, check_finite=False)
#alpha = cho_solve(cF, y_train, check_finite=False)





#Get the data in the form (T,t,K) and prices 
#data = pd.DataFrame({"Expiry":df.Expiry,"Tenor":df.Tenor})
#Kr = K.reset_index(drop = True)
#data[Kr.columns] = Kr[Kr.columns]
#data = data.melt(id_vars=['Expiry','Tenor'],var_name= 'Relative_Strike', value_name='Absolut_strike' ) 
#Swaptions = np.array(prices).T.ravel() 
#data = data.drop(columns = ['Relative_Strike']) 



#X_obs = np.column_stack((expiries.values,tenors,K.iloc[:,4].values))
#y_obs = prices.iloc[:,4]
#model = ConstrainedSwaptionGPCube(
#        T_grid=expiries.values,
#        t_grid=np.unique(tenors), 
#        K_grid=K.iloc[0,1:].values,
#        theta_T=0.3,
#        theta_t=0.3,
#        theta_K=0.3,
#        sigma=1.0,
#        noise=0.01,
#        kind='payer'
#    ) 
#model.fit_mle(y_obs, X_obs, restarts=10, seed=0)
#k = model.kernel
#model.fit_map(y_obs, X_obs, year_step_inplane=True) 
#pred = model.predict(X_obs[:4])   

    
            
    
    

