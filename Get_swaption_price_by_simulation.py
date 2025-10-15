# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 09:51:29 2025

@author: Dell
"""

from Black_Swaption_Price import Swaption
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d




# Get the  vols data
df = pd.read_excel(r"C:\Users\Dell\Documents\ML_Project\GP_Regression\market_data.xlsx",skiprows=1) 
df = df.iloc[:-1]

# Arrange by Expiry 
df = df.sort_values(["Expiry", "Tenor"]).reset_index(drop=True)
cols = [ 'Expiry', 'Tenor' ,'Fwd', -150, -100, -50, -25, 0, 25, 50, 100, 150]
df = df[cols]

# Take the strikes and convert them to percentage 
spreads = df.columns.difference(['Expiry','Tenor','Fwd']).tolist()
F = df.Fwd.tolist()
Expiries = df['Expiry'].tolist()
Tenors = df['Tenor'].tolist()
K = pd.DataFrame(index = F, columns = spreads)
K.index.name = 'Fwd'

for spread in spreads :
    for f in F :
        K.loc[f,spread] = f + spread*0.0001
        
# Get the vols 
vols = df.loc[:,spreads]      

# Plot the curves         
fig, ax = plt.subplots(7,5, figsize = (20,18), squeeze = True, gridspec_kw={'hspace':0.5, 'wspace':0.5})
ax = ax.flatten()     
for i in range(len(F)):
    ax[i].set_title(f'Expiry {df.Expiry[i]} Tenor {df.Tenor[i]}')
    ax[i].scatter(K.iloc[i,:],vols.iloc[i,:])
    ax[i].plot(K.iloc[i,:],vols.iloc[i,:])
plt.show()    

# Get the swap rate 
swap_rate = pd.read_csv(r'C:\Users\Dell\Documents\ML_Project\GP_Regression\Swap_curve.txt')

#Interpolate the swap curve 
maturities_to_interpolate = np.linspace(1,30,30)
f_interpolation = interp1d(swap_rate.Tenor, swap_rate.Rate, kind='linear', fill_value="extrapolate")
swap_rate_interpolated = f_interpolation(maturities_to_interpolate)
#Swap rate curve interploated 
swap_curve_interpolated = pd.DataFrame({'Tenor' : maturities_to_interpolate, 'Rate' :swap_rate_interpolated })  
# Plot the swap curve 
plt.figure(figsize=(16,14))
plt.scatter(swap_rate.Tenor, swap_rate.Rate, c='red', label ='Before interpolation')
plt.plot(maturities_to_interpolate, swap_rate_interpolated, label = 'After interpolation')
#plt.plot(swap_rate.Tenor,swap_rate.Rate, marker='o')
plt.xlabel('Maturity')
plt.ylabel('Swap rate')
plt.xticks(df.Tenor)
plt.title('Swap rate ')
plt.legend()
plt.show()

# Deduce the yield curve from the swap curve
# Algorithme de boostraping 




            
       
# Calculate the price of all swaption 
swaption_price_curve = pd.DataFrame(columns = df.columns)
swaption_price_curve[['Expiry','Tenor','Fwd']] = df[['Expiry','Tenor','Fwd']]
for spread in spreads :
    liste = []
    for i in range(len(F)):
    #    if spread < 0 :
            swaption_ob = Swaption(F[i], Expiries[i], Tenors[i], K.loc[:,spread].values[i], vols.loc[i,spread], swap_curve_interpolated, 'payer')
            liste.append(swaption_ob.price_with_BS())
     #   else :
      #      swaption_ob = Swaption(F[i], Expiries[i], Tenors[i], K.loc[:,spread].values[i], vols.loc[i,spread], swap_curve_interpolated, 'payer')
       #     liste.append(swaption_ob.price_with_BS())
            
    
    swaption_price_curve[spread] = liste
    
    del liste[:]   
    
        
# Look the form of the swaptions prices curve 
fig1, ax1 = plt.subplots(7,5,figsize=(20,18),squeeze = True, gridspec_kw={'hspace':0.5, 'wspace':0.5} )
ax1 = ax1.flatten()
for i in range(len(F)):
        ax1[i].set_title(f'Expiry {df.Expiry[i]} Tenor {df.Tenor[i]}')
        ax1[i].scatter(K.iloc[i,:],swaption_price_curve.loc[i,spreads] , c='red')
        ax1[i].plot(K.iloc[i,:],swaption_price_curve.loc[i,spreads] , c='black')
        
plt.show()        
        
#Convert to excel 
swaption_price_curve.to_csv("Swaption_price_simulated.txt",index = False)       

        
    
     
    
    