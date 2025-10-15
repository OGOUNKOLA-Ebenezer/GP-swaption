# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 08:11:04 2025

@author: Dell
"""
import numpy as np
from scipy.stats import norm
class Swaption():
    """
    Calculate the price of a swaption given parameters and the swap yiel curve 
    """
    def __init__(self, F, T, t, K, vol, swap_curve,type='payer'):
        
        self.F = F
        self.T = T
        self.K = K
        self.vol = vol 
        self.swap_curve = swap_curve.copy()
        self.t = t
        self.type = type
        
        self.swap_curve.columns = ['tenors', 'swap_rate']
        
    def boostrap_discount_factor_from_swap_rate(self,freq=1):
        tau = 1/freq 
        try :
            index_t = self.swap_curve['tenors'].tolist().index(self.t)
            swap_rate_t = self.swap_curve['swap_rate'].values[:index_t+1]
            discount_factor_t = []
            df1 = 1/(1+tau*swap_rate_t[0])
            if index_t == 0:
                return df1
            
            discount_factor_t.append(df1) 
            for i in range(1,index_t+1):
                sum_d = sum(discount_factor_t)
                dfi = (tau - swap_rate_t[i]*sum_d ) / (swap_rate_t[i] + tau)
                discount_factor_t.append(dfi)
            
            return discount_factor_t  
        
        except ValueError :
            
                
            print(f'Le tenor t = {self.t} est pas dans la liste des tenors de la courbe des taux! Veuillez entrez un tenor existant')
                
            
    def annuity(self, freq=1):
        tau = 1/freq
        l = self.boostrap_discount_factor_from_swap_rate(freq)
        return sum([tau*i for i in l ])
    
    
    def price_with_BS(self):
        annuity = self.annuity()
        if self.vol == 0 or self.T == 0:
            if self.type =='payer':
                return annuity * max(self.F - self.K,0)
            else:
                return annuity * max(self.K - self.F,0) 
            
        if self.F <= 0 or self.K <= 0:
            return annuity * max((self.F -self.K) if self.type == 'payer' else (self.K - self.K), 0)
            
        d1 = (np.log(self.F / self.K) + (self.vol**2 / 2) * self.T) / (self.vol * np.sqrt(self.T))
        d2 = (np.log(self.F / self.K) + ( - self.vol**2 / 2) * self.T) / (self.vol * np.sqrt(self.T))
            
        if self.type == 'payer': 
            price = annuity * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        else:  # receiver = put
            price = annuity * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        return price  
            
