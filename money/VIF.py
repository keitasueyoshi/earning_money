# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def calculate_and_plot_vif(df_name, selected_features=None, sort=True, threshold_lines=True, add_constant=True, drop_const=True):
    """
    指定したDataFrameから、指定したfeatureだけでVIFを計算し、横棒グラフで出力する関数(モデルの考察や重要変数の関係性を確認する用)
    from VIF import calculate_and_plot_vif
    selected_features =['',']
    vif_result = calculate_and_plot_vif(df, selected_features=selected_features)
    print(vif_result)
    """

    # 必要なカラムだけ選択
    if selected_features is not None:
        X = df_name[selected_features]
    else:
        X = df_name.copy()

    # 定数項追加（オプション）
    if add_constant:
        X = sm.add_constant(X)

    # VIF計算
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # const列を削除する（オプション）
    if drop_const:
        vif_data = vif_data[vif_data["feature"] != "const"]

    # ソート
    if sort:
        vif_data = vif_data.sort_values(by="VIF", ascending=False)

    # --- 棒グラフプロット ---
    plt.figure(figsize=(10, 6))
    bars = plt.barh(vif_data['feature'], vif_data['VIF'], color='orange')
    plt.xlabel('VIF Value')
    plt.title('Variance Inflation Factor (VIF) per Feature')

    # 各棒の右横にVIFの値を表示
    for bar, value in zip(bars, vif_data['VIF']):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{value:.2f}', va='center', fontsize=10)

    # VIF=5とVIF=10の閾値線
    if threshold_lines:
        plt.axvline(x=5, color='red', linestyle='--', label='VIF=5 Threshold')
        plt.axvline(x=10, color='orange', linestyle='--', label='VIF=10 Threshold')
        plt.legend()

    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()

    return vif_data

