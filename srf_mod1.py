# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:30:21 2025

@author: SYSU_Yuqiang
@email:2942204121@qq.com

"""

import os
import glob
import csv
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def read_modtran_tape7(filename):
    """读取 MODTRAN 的 tape7 文件，提取波数和辐射值"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith('***'):
                continue
            try:
                values = list(map(float, line.strip().split()))
                if len(values) >= 9:
                    data.append([values[0], values[8]])  
            except:
                continue
    
    if not data:
        raise ValueError("未找到有效的辐射数据")
    
    df = pd.DataFrame(data, columns=["wavenumber", "radiance"])
    df = df.sort_values(by="wavenumber")
    df["radiance"] = df["radiance"] * 1e8  # 单位转换
    return df["wavenumber"].values, df["radiance"].values

def read_modis_srf(srf_path):
    """读取 MODIS 的 SRF 文件并归一化"""
    with open(srf_path, "r") as file:
        lines = file.readlines()
    
    start_idx = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                start_idx = i
                break
            except ValueError:
                continue
    
    if start_idx is None:
        raise ValueError("未找到有效的SRF数据行")
    
    srf_df = pd.read_csv(
        srf_path,
        delim_whitespace=True,
        skiprows=start_idx,
        names=["wavenumber", "response"],
        dtype={"wavenumber": float, "response": float}
    )
    
    srf_wavenum = srf_df["wavenumber"].values
    srf_response = srf_df["response"].values
    srf_response = srf_response / np.max(srf_response)  
    return srf_wavenum, srf_response

def convolve_radiance(modtran_wavenum, modtran_radiance, srf_wavenum, srf_response):
    """计算归一化辐亮度"""
    valid = (modtran_wavenum >= srf_wavenum.min()) & (modtran_wavenum <= srf_wavenum.max())
    wavenum = modtran_wavenum[valid]
    radiance = modtran_radiance[valid]
    
    if len(wavenum) == 0:
        raise ValueError("SRF波数范围与MODTRAN数据无重叠")
    
    interp_srf = interp1d(
        srf_wavenum, srf_response,
        kind='linear', bounds_error=False, fill_value=0.0
    )
    srf_interp = interp_srf(wavenum)
    
    numerator = np.trapz(radiance * srf_interp, wavenum)
    denominator = np.trapz(srf_interp, wavenum)
    
    if denominator == 0:
        raise ValueError("SRF积分值为零，波数范围可能不匹配")
    
    return numerator / denominator

def process_folder(folder_path, srf_data):
    """处理单个MODTRAN结果文件夹，返回各SRF的卷积结果"""
    tape7_path = os.path.join(folder_path, "tape7")
    if not os.path.exists(tape7_path):
        raise FileNotFoundError(f"未找到tape7文件: {tape7_path}")
    
    wavenum, radiance = read_modtran_tape7(tape7_path)
    results = {}
    for srf_name, srf_wavenum, srf_response in srf_data:
        try:
            norm_rad = convolve_radiance(wavenum, radiance, srf_wavenum, srf_response)
            results[srf_name] = norm_rad
        except Exception as e:
            print(f"卷积失败 {srf_name}: {str(e)}")
            results[srf_name] = np.nan
    return results

def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    params = {}
    
    p_part = parts[0]
    if p_part.startswith('P'):
        p_str = p_part[1:]
        params['p_number'] = int(p_str) if p_str.isdigit() else p_str
    else:
        params['p_number'] = 'Unknown'
    
    angle_params = []
    for part in parts:
        if part.lower().startswith('angle'):
            match = re.search(r'angle_?([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)', folder_name)
            if match:
                angle_params = list(map(float, match.groups()))
            break

    if len(angle_params) != 4:
        print(f"警告: 解析角度失败, 文件夹名: {folder_name}")
        angle_params = [0.0, 0.0, 0.0, 0.0]

    params.update({
        'vza': angle_params[0],
        'vaa': angle_params[1],
        'sza': angle_params[2],
        'saa': angle_params[3]
    })
    return params

def main():
    srf_config = [
        ("modis", "D:/example/srf/aqua/rtcoef_eos_2_modis-C7_srf_ch03.txt"),
        ("mersi", "D:/example/srf/fy3d/rtcoef_fy3_4_mersi2_srf_ch01.txt")
    ]
    
    
    srf_data = []
    for name, path in srf_config:
        try:
            srf_wavenum, srf_response = read_modis_srf(path)
            srf_data.append( (name, srf_wavenum, srf_response) )
            print(f"成功读取SRF: {name}")
        except Exception as e:
            print(f"读取SRF {name} 失败: {e}")
            raise
    
    base_dir = r"F:/modtran5/MODTRAN_Sunny3"
    output_csv = "D:/example/srf/modtran_dual_srf_comparison.csv"
    
    folder_info = []
    for folder_path in glob.glob(os.path.join(base_dir, "*")):
        if os.path.isdir(folder_path):
            folder_name = os.path.basename(folder_path)
            params = parse_folder_name(folder_name)
            
            try:
                p_num = int(params['p_number'])
            except (ValueError, TypeError):
                p_num = float('inf')
            
            angle_params = (
                params['vza'],
                params['vaa'],
                params['sza'],
                params['saa']
            )
            
            folder_info.append( (p_num, *angle_params, folder_path) )
    
    folder_info.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Profile", "VZA", "VAA", "SZA", "SAA", "modis_rad", "mersi_rad"])
        
        for item in folder_info:
            p_num, vza, vaa, sza, saa, folder_path = item
            folder_name = os.path.basename(folder_path)
            
            try:
                rad_values = process_folder(folder_path, srf_data)
                modis_rad = rad_values.get('modis', np.nan)
                mersi_rad = rad_values.get('mersi', np.nan)
                
                writer.writerow([
                    f"P{int(p_num) if not isinstance(p_num, str) else p_num}",
                    vza,
                    vaa,
                    sza,
                    saa,
                    f"{modis_rad:.6f}",
                    f"{mersi_rad:.6f}"
                ])
                print(f"成功处理: {folder_name}")
            except Exception as e:
                print(f"处理失败 {folder_name}: {str(e)}")

if __name__ == "__main__":
    main()