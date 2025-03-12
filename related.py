# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:38:43 2025

@author: SYSU_Yuqiang
@email:2942204121@qq.com

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def read_csv(file_path):
    return pd.read_csv(file_path, delimiter= None, engine='python')  

def linear_regression(input_file, indices):
    df = read_csv(input_file)
    
    df = df.iloc[indices, :]
    
    df = df.dropna(subset=['modis_rad', 'mersi_rad'])
    
    X = df['modis_rad'].values.reshape(-1,1)  
    y = df['mersi_rad'].values.reshape(-1,1)  

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred) 
    #r2 = r2 - 0.0142  
    
    plt.figure(figsize=(12,12), dpi=300)
    plt.scatter(X, y, color='r', s=80, label='Observation')
    plt.plot(X, y_pred, color='black', lw=3, label='Regression Line')
    
    plt.xlabel('MODIS Rad', fontsize=24)
    plt.ylabel('MERSI-II Rad', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    formula = (
        f"y = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}\n"
        f"R² = {r2:.5f}"
    )
    plt.text(max(X) * 1,
             min(y) * 1, 
             formula,
             fontsize=20,
             color='black',
             ha='right'
    )
    
    plt.legend(fontsize=20, markerscale=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

    print(f"回归系数: {model.coef_[0][0]:.6f}")
    print(f"截距: {model.intercept_[0]:.6f}")
    print(f"R²: {r2:.6f}")

if __name__ == "__main__":
    input_file = "D:/example/srf/modtran_dual_srf_comparison.csv"
    selected_indices = range(0,3600)  
    
    linear_regression(input_file, selected_indices)
