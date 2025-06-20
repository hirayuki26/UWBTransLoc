import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NO_SIGNAL_VALUE = -100.0 # Changed Null Value
QUANTITATIVE_COLUMNS = ['x', 'y'] # Regression Columns

class WiFiDataset(Dataset):
    def __init__(self, data_df, rss_cols, loc_cols, transform=None, target_transform=None):
        # ここではもうファイルパスではなく、整形済みのDataFrameを受け取る
        self.rss_data = data_df[rss_cols].values
        self.loc_data = data_df[loc_cols].values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.rss_data)

    def __getitem__(self, idx):
        rss = self.rss_data[idx]
        loc = self.loc_data[idx]

        if self.transform:
            rss = self.transform(rss)
        if self.target_transform:
            loc = self.target_transform(loc)
        
        # Convert to float32 for PyTorch compatibility
        return torch.tensor(rss, dtype=torch.float32), torch.tensor(loc, dtype=torch.float32)

def align_dataframe_columns(df, target_cols, fill_value=NO_SIGNAL_VALUE):
    """
    DataFrameの列をtarget_colsに合わせ、足りない列はfill_valueで埋める。
    """
    current_cols = set(df.columns)
    missing_cols = [col for col in target_cols if col not in current_cols]
    
    # 足りない列を追加し、fill_valueで埋める
    for col in missing_cols:
        df[col] = fill_value
    
    # target_colsの順序に並べ替える (重要)
    return df[target_cols]

def create_dataloaders(source_train_path, target_train_path, target_test_path):
    loc_cols = QUANTITATIVE_COLUMNS

    # 1. 全てのデータファイルを読み込み、全てのAP列名を収集する
    source_train_raw = pd.read_csv(source_train_path)
    target_train_raw = pd.read_csv(target_train_path)
    target_test_raw = pd.read_csv(target_test_path)

    # 全てのデータフレームから位置情報列を除くすべての列名（APのMACアドレス）を収集
    all_ap_cols_set = set()
    for df in [source_train_raw, target_train_raw, target_test_raw]:
        current_ap_cols = [col for col in df.columns if col not in loc_cols]
        all_ap_cols_set.update(current_ap_cols)
    
    all_rss_cols = sorted(list(all_ap_cols_set)) # APの列名をソートして固定順にする (重要)

    # 2. 各データセットを整形する (AP列の統一と欠損値の埋め合わせ)
    source_train_aligned = align_dataframe_columns(source_train_raw.copy(), all_rss_cols + loc_cols)
    target_train_aligned = align_dataframe_columns(target_train_raw.copy(), all_rss_cols + loc_cols)
    target_test_aligned = align_dataframe_columns(target_test_raw.copy(), all_rss_cols + loc_cols)

    # Load data for scaling (整形後のデータを使用)
    # Concatenate all RSS data for a comprehensive scaling range
    all_rss_data = pd.concat([
        source_train_aligned[all_rss_cols],
        target_train_aligned[all_rss_cols],
        target_test_aligned[all_rss_cols]
    ])
    
    # Concatenate all location data for scaling
    all_loc_data = pd.concat([
        source_train_aligned[loc_cols],
        target_train_aligned[loc_cols],
        target_test_aligned[loc_cols]
    ])

    # Initialize scalers
    rss_scaler = MinMaxScaler(feature_range=(-1, 1)) # Normalize RSS to -1 to 1, common for GANs
    loc_scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize locations to 0 to 1

    # Fit scalers on combined data
    rss_scaler.fit(all_rss_data)
    loc_scaler.fit(all_loc_data)

    # Define transforms
    # スケーラーは2D配列を期待するので、入力も2Dにする (x.reshape(1, -1))
    # 出力は1Dに戻す (.flatten())
    rss_transform = lambda x: rss_scaler.transform(x.reshape(1, -1)).flatten()
    loc_transform = lambda y: loc_scaler.transform(y.reshape(1, -1)).flatten()

    # Create datasets (整形済みのDataFrameを渡す)
    source_train_dataset = WiFiDataset(source_train_aligned, all_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    target_train_dataset = WiFiDataset(target_train_aligned, all_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    target_test_dataset = WiFiDataset(target_test_aligned, all_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)

    # Create dataloaders
    batch_size = 64 # You can adjust this
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    # Store scalers for inverse transformation later if needed
    data_scalers = {
        'rss_scaler': rss_scaler,
        'loc_scaler': loc_scaler,
        'rss_cols': all_rss_cols, # ここをall_rss_colsに変更
        'loc_cols': loc_cols
    }

    return source_train_loader, target_train_loader, target_test_loader, data_scalers

# Usage (変更なし):
place_name = 'OfficeP2'
source_train_path = f'./data/{place_name}/csv/{place_name}_1_training.csv'
target_train_path = f'./data/{place_name}/csv/{place_name}_2_training.csv'
target_test_path = f'./data/{place_name}/csv/{place_name}_2_testing.csv'

source_train_loader, target_train_loader, target_test_loader, data_scalers = create_dataloaders(
    source_train_path, target_train_path, target_test_path
)

# Get input dimension for network architecture
sample_rss, _ = next(iter(source_train_loader))
input_dim = sample_rss.shape[1] # このinput_dimがall_rss_colsの数になる
output_dim = 2 # X, Y coordinates

print(f"Input Dimension (RSS features): {input_dim}")
print(f"Output Dimension (Location features): {output_dim}")
print(f"Number of samples in source_train: {len(source_train_loader.dataset)}")
print(f"Number of samples in target_train: {len(target_train_loader.dataset)}")
print(f"Number of samples in target_test: {len(target_test_loader.dataset)}")