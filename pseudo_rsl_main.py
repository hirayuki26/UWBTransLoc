import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import random
import copy
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NO_SIGNAL_VALUE = -100.0 # Changed Null Value
QUANTITATIVE_COLUMNS = ['x', 'y'] # Regression Columns

# --- 1. データセットの準備とデータローダーの定義 ---
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

def create_dataloaders(source_train_path, target_train_path, target_test_path, ap_filter_list=None):
    loc_cols = QUANTITATIVE_COLUMNS

    # ap_filter_list が None の場合は、デフォルトで空のリスト（全てのAPを使用）とする
    if ap_filter_list is None:
        ap_filter_list = []

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

    # rsl_only_cols = all_rss_cols
    # rsl_only_cols = [col for col in all_rss_cols if col.endswith('_rsl')]
    rsl_only_cols = [col for col in all_rss_cols if col.endswith('_rng_rng')]

    # APフィルタリング
    if ap_filter_list:
        # ap_filter_list に要素がある場合（特定のAPを指定した場合）
        final_rss_cols = []
        for col in rsl_only_cols:
            # col が ap_filter_list のいずれかの要素で始まるかチェックする
            if any(col.startswith(ap) for ap in ap_filter_list):
                final_rss_cols.append(col)
    else:
        # ap_filter_list が空のリストの場合（全てのAPを指定したい場合）
        # rsl_only_cols の内容をそのまま使用する
        final_rss_cols = rsl_only_cols

    # 2. 各データセットを整形する (AP列の統一と欠損値の埋め合わせ)
    source_train_aligned = align_dataframe_columns(source_train_raw.copy(), all_rss_cols + loc_cols)
    target_train_aligned = align_dataframe_columns(target_train_raw.copy(), all_rss_cols + loc_cols)
    target_test_aligned = align_dataframe_columns(target_test_raw.copy(), all_rss_cols + loc_cols)

    # Load data for scaling (整形後のデータを使用)
    # Concatenate all RSS data for a comprehensive scaling range
    # all_rss_data = pd.concat([
    #     source_train_aligned[rsl_only_cols],
    #     target_train_aligned[rsl_only_cols],
    #     target_test_aligned[rsl_only_cols]
    # ])
    # Load data for scaling
    all_rss_data = pd.concat([
        source_train_aligned[final_rss_cols], 
        target_train_aligned[final_rss_cols],
        target_test_aligned[final_rss_cols]
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
    # source_train_dataset = WiFiDataset(source_train_aligned, rsl_only_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    # target_train_dataset = WiFiDataset(target_train_aligned, rsl_only_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    # target_test_dataset = WiFiDataset(target_test_aligned, rsl_only_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    source_train_dataset = WiFiDataset(source_train_aligned, final_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    target_train_dataset = WiFiDataset(target_train_aligned, final_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    target_test_dataset = WiFiDataset(target_test_aligned, final_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)

    # Create dataloaders
    batch_size = 64 # You can adjust this
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    # 追加: ターゲット訓練データセットの各座標ごとのデータ数を計算
    target_train_loc_counts = {}
    # スケール前の生の座標を使用するため、loc_scaler.inverse_transform を使用
    aa_scaled_locs = np.array([target_train_dataset[i][1].numpy() for i in range(len(target_train_dataset))])
    temp_target_loc_unscaled = loc_scaler.inverse_transform(aa_scaled_locs)
    for loc_array in temp_target_loc_unscaled:
        loc_tuple = tuple(loc_array) # リストは辞書のキーになれないためタプルに変換
        target_train_loc_counts[loc_tuple] = target_train_loc_counts.get(loc_tuple, 0) + 1
    print(target_train_loc_counts)

    # Store scalers for inverse transformation later if needed
    data_scalers = {
        'rss_scaler': rss_scaler,
        'loc_scaler': loc_scaler,
        # 'rss_cols': rsl_only_cols, # ここをall_rss_colsに変更
        'rss_cols': final_rss_cols, # APフィルタリング
        'loc_cols': loc_cols,
        'target_train_loc_counts': target_train_loc_counts # 追加: 各座標のデータ総数を格納
    }

    return source_train_loader, target_train_loader, target_test_loader, data_scalers

# --- 2. TransLocのネットワークアーキテクチャ定義 ---

# スペクトル正規化のラッパー関数 
def spectral_norm(module):
    return torch.nn.utils.spectral_norm(module)

# ConvBlk (CNN Block) 
class ConvBlk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) # 
        self.relu = nn.LeakyReLU(0.1) # 
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# DeConvBlk (Deconvolutional Block) 
class DeConvBlk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(DeConvBlk, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding) # 
        self.bn = nn.BatchNorm2d(out_channels) # 
        self.relu = nn.LeakyReLU(0.1) # 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # 

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

# Feature Extractor 
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, z_dim=16): # z_dim is latent feature dimension 
        super(FeatureExtractor, self).__init__()
        self.fc_initial = nn.Linear(input_dim, 1024) # 
        # Reshape to 32x32 for CNNs 
        self.conv_blk1 = ConvBlk(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2) # 5x5x16 
        self.conv_blk2 = ConvBlk(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) # 5x5x32 
        # Final FC layer to get z_dim 
        self.fc_final = nn.Linear(32 * 8 * 8, z_dim) # Output size after two ConvBlk (32*32 -> 16*16 -> 8*8)

    def forward(self, x):
        x = self.fc_initial(x)
        x = x.view(-1, 1, 32, 32) # Reshape to (batch_size, channels, height, width)
        x = self.conv_blk1(x)
        x = self.conv_blk2(x)
        x = x.view(x.size(0), -1) # Flatten for FC layer
        z = self.fc_final(x)
        return z

# Generator 
class Generator(nn.Module):
    def __init__(self, z_dim=16, output_dim=1024, domain_dim=1): 
        super(Generator, self).__init__()
        # Z (16) + Domain (1) input for FC layer initially 
        self.fc_initial = nn.Linear(z_dim + domain_dim, 2048) # 
        
        self.deconv_blk1_input_channels = 32 # Assuming 2048 is reshaped to 8x8x32
        self.deconv_blk1 = DeConvBlk(self.deconv_blk1_input_channels, 16, kernel_size=5, stride=1, padding=2) # 5x5x16 
        self.deconv_blk2 = DeConvBlk(16, 1, kernel_size=5, stride=1, padding=2) # 5x5x1 
        
        # Final layer to match the original input dimension (1024 as vector)
        self.final_conv_to_vector = nn.Conv2d(1, 1, kernel_size=1) 
        self.fc_final_output = nn.Linear(32*32*1, output_dim) 

    def forward(self, z, domain_label):
        if domain_label.dim() == 1:
            domain_label = domain_label.unsqueeze(1) 
        
        combined_input = torch.cat([z, domain_label], dim=1)

        x = self.fc_initial(combined_input)
        
        # Reshape for DCNN layers. Assuming 8x8 spatial dimensions.
        spatial_dim_start = int(math.sqrt(x.size(1) / self.deconv_blk1_input_channels))
        if spatial_dim_start * spatial_dim_start * self.deconv_blk1_input_channels != x.size(1):
            spatial_dim_start = 8 # Fallback if calculation is not perfect
            
        x = x.view(x.size(0), self.deconv_blk1_input_channels, spatial_dim_start, spatial_dim_start) 
        
        x = self.deconv_blk1(x)
        x = self.deconv_blk2(x) 
        
        # Final output transformation to match the original input vector dimension
        x = self.final_conv_to_vector(x)
        reconstructed_fingerprint = self.fc_final_output(x.view(x.size(0), -1))
        
        return reconstructed_fingerprint

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim, domain_dim=1):
        super(Discriminator, self).__init__()
        self.fc_initial = nn.Linear(input_dim, 1024)
        
        self.conv_blk1 = ConvBlk(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2) # 5x5x8 
        self.conv_blk2 = ConvBlk(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2) # 5x5x16 

        self.common_fc_output_dim = 16 * 8 * 8 

        # Discriminator for source domain (D_s) 
        self.fc_ds1 = spectral_norm(nn.Linear(self.common_fc_output_dim, 256)) 
        self.fc_ds2 = spectral_norm(nn.Linear(256, 128))
        self.fc_ds3 = spectral_norm(nn.Linear(128, 32))
        self.fc_ds4 = spectral_norm(nn.Linear(32, 1)) # Binary output for real/fake (source)
        
        # Discriminator for target domain (D_t) 
        self.fc_dt1 = spectral_norm(nn.Linear(self.common_fc_output_dim, 256))
        self.fc_dt2 = spectral_norm(nn.Linear(256, 128))
        self.fc_dt3 = spectral_norm(nn.Linear(128, 32))
        self.fc_dt4 = spectral_norm(nn.Linear(32, 1)) # Binary output for real/fake (target)

    def forward(self, x):
        x = self.fc_initial(x)
        x = x.view(-1, 1, 32, 32)
        x = self.conv_blk1(x)
        x = self.conv_blk2(x)
        x = x.view(x.size(0), -1) 

        # Source Discriminator branch
        ds = self.fc_ds1(x)
        ds = self.fc_ds2(ds)
        ds = self.fc_ds3(ds)
        ds_output = self.fc_ds4(ds) 

        # Target Discriminator branch
        dt = self.fc_dt1(x)
        dt = self.fc_dt2(dt)
        dt = self.fc_dt3(dt)
        dt_output = self.fc_dt4(dt) 

        return ds_output, dt_output

# Location Predictor 
class LocationPredictor(nn.Module):
    def __init__(self, z_dim=16, output_loc_dim=2):
        super(LocationPredictor, self).__init__()
        # R_c (shared module) 
        self.rc_fc1 = nn.Linear(z_dim, 1024)
        self.rc_conv_blk = ConvBlk(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2) # 5x5x16
        self.rc_fc2 = nn.Linear(16 * 16 * 16, 2048) 

        # R_1, R_2, R_3 (parallel sub-modules) 
        self.r1_conv_blk = ConvBlk(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2) 
        self.r1_fc1 = nn.Linear(16 * 4 * 4, 2048)
        self.r1_fc2 = nn.Linear(2048, output_loc_dim) # Output (X, Y) 

        self.r2_conv_blk = ConvBlk(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.r2_fc1 = nn.Linear(16 * 4 * 4, 2048)
        self.r2_fc2 = nn.Linear(2048, output_loc_dim)

        self.r3_conv_blk = ConvBlk(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.r3_fc1 = nn.Linear(16 * 4 * 4, 2048)
        self.r3_fc2 = nn.Linear(2048, output_loc_dim)

    def forward(self, z):
        # R_c (shared module)
        rc_features = self.rc_fc1(z)
        rc_features = rc_features.view(-1, 1, 32, 32) 
        rc_features = self.rc_conv_blk(rc_features)
        rc_features = rc_features.view(rc_features.size(0), -1) 
        rc_features = self.rc_fc2(rc_features)
        
        # Reshape rc_features for conv_blk input of R1, R2, R3
        rc_features_reshaped = rc_features.view(rc_features.size(0), 32, 8, 8)

        # R_1 branch
        r1_out = self.r1_conv_blk(rc_features_reshaped)
        r1_out = r1_out.view(r1_out.size(0), -1)
        r1_out = self.r1_fc1(r1_out)
        r1_pred = r1_out # No activation after final fc_final 
        r1_pred = self.r1_fc2(r1_pred)

        # R_2 branch
        r2_out = self.r2_conv_blk(rc_features_reshaped)
        r2_out = r2_out.view(r2_out.size(0), -1)
        r2_out = self.r2_fc1(r2_out)
        r2_pred = r2_out # No activation after final fc_final 
        r2_pred = self.r2_fc2(r2_pred)

        # R_3 branch
        r3_out = self.r3_conv_blk(rc_features_reshaped)
        r3_out = r3_out.view(r3_out.size(0), -1)
        r3_out = self.r3_fc1(r3_out)
        r3_pred = r3_out # No activation after final fc_final 
        r3_pred = self.r3_fc2(r3_pred)
        
        # Average of predictions for final output (Eq 9 in TransLoc paper) 
        avg_pred = (r1_pred + r2_pred + r3_pred) / 3

        return avg_pred, r1_pred, r2_pred, r3_pred

# TransLocモデル全体の定義 
class TransLoc(nn.Module):
    def __init__(self, input_dim, z_dim, output_loc_dim, domain_dim=1):
        super(TransLoc, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, z_dim).to(device)
        self.generator = Generator(z_dim, input_dim, domain_dim).to(device) 
        self.discriminator = Discriminator(input_dim, domain_dim).to(device)
        self.location_predictor = LocationPredictor(z_dim, output_loc_dim).to(device)

    def forward(self, x, domain_label):
        # 個々のコンポーネントの呼び出しはトレーニングループ内で行われます。
        pass


# --- 3. 損失関数の定義 ---

# 再構築損失 (L_GE) 
# Mean Squared Error (MSE) を使用
reconstruction_criterion = nn.MSELoss() 

# 識別器の損失 (L_D) - Least Squares GANs (LSGANs) の損失を使用 
# 論文ではb=1, a=0, c=1 (Eq 9) or b=1, a=-1, c=0 (Eq 8)が提案されているが、ここではBCEWithLogitsLossを使う場合のLSGANsの一般的な解釈に基づく。
# PyTorchではnn.MSELoss()と組み合わせることが多い。
# D(x) -> real, D(G(z)) -> fake
# For D: min 0.5 * (D(x) - 1)^2 + 0.5 * (D(G(z)) - 0)^2
# For G: min 0.5 * (D(G(z)) - 1)^2
criterion_discriminator = nn.MSELoss()
criterion_generator_adv = nn.MSELoss()

# 位置予測器の損失 (L_R) 
# MSE (Regression task) 
location_criterion = nn.MSELoss()

# サイクル一貫性損失 (L_CC) 
# L1 Loss (MAE) for feature level and prediction level consistency
cycle_consistency_criterion_f = nn.L1Loss() # 
cycle_consistency_criterion_p = nn.MSELoss() #  (距離の二乗和)

# --- 4. 学習戦略とトレーニングループ ---

def train_transloc(model, source_train_loader, target_train_loader, target_test_loader, data_scalers,
                   num_epochs=100, lr_fe_lp=0.0001, lr_g=0.0002, lr_d=0.0002,
                   lambda_D=1, lambda_R=1, lambda_CC=1, epsilon_tri_net=1e-4):
    
    # オプティマイザの定義 
    optimizer_fe = optim.Adam(model.feature_extractor.parameters(), lr=lr_fe_lp, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_lp = optim.Adam(model.location_predictor.parameters(), lr=lr_fe_lp, betas=(0.5, 0.999))

    # ハイパーパラメータeta (η) - 勾配反転層に相当する制御 
    # 論文ではeta=10がデフォルト 
    eta_gradient_reversal = 10 

    # Loss history lists (追加: 損失履歴を記録するためのリストを初期化)
    total_loss_d_history = []
    total_loss_g_adv_history = []
    total_loss_g_rec_history = []
    total_loss_lp_history = []
    total_loss_cc_f_history = []
    total_loss_cc_p_history = []
    test_error_history = []
    test_error_epochs_recorded = []

    # New histories for pseudo-label metrics
    pseudo_label_counts_history = []
    pseudo_label_consistency_error_history = []
    # 追加: 擬似ラベルと真のラベルの誤差履歴
    pseudo_label_true_error_history = [] 

    # 追加: エポックごとの擬似ラベルと元のターゲット座標の情報を記憶するリスト
    # 形式: [{epoch1_data}, {epoch2_data}, ...]
    # epochX_data: { (target_x, target_y): [pseudo_label1, pseudo_label2, ...], ... }
    all_epochs_pseudo_label_locations = [] 
    # 追加: エポックごとの詳細な擬似ラベル統計を記録するリスト
    # 形式: [{epoch1_stats}, {epoch2_stats}, ...]
    # epochX_stats: { (target_x, target_y): {'count': N, 'ratio': R, 'avg_true_error': E}, ... }
    all_epochs_detailed_pseudo_label_stats = [] 

    # --- 追加: 最後のEpochの擬似ラベルとRSLデータを収集するリスト ---
    # 形式: [(RSL_features_tensor, Pseudo_label_tensor)]
    last_epoch_pseudo_data = []
    
    # 事前に計算した各ターゲット座標のデータ総数を取得
    target_train_total_counts = data_scalers['target_train_loc_counts']

    # 事前学習フェーズ 
    print("--- Pre-training Phase ---")
    pretrain_epochs = 50 # Adjust based on complexity and convergence
    for epoch in range(pretrain_epochs):
        model.train()
        total_reconstruction_loss = 0
        total_source_lp_loss = 0

        for i, (source_rss, source_loc) in enumerate(source_train_loader):
            source_rss, source_loc = source_rss.to(device), source_loc.to(device)

            # Feature ExtractorとGeneratorの事前学習 (L_GE) 
            optimizer_fe.zero_grad()
            optimizer_g.zero_grad()
            
            # Source DomainのZを抽出
            z_s = model.feature_extractor(source_rss)
            
            # GeneratorでSource RSSを再構築
            # ドメインラベルはSource (0) と仮定
            source_domain_label = torch.zeros(source_rss.size(0), 1).to(device)
            reconstructed_source_rss = model.generator(z_s, source_domain_label)
            
            loss_ge = reconstruction_criterion(reconstructed_source_rss, source_rss)
            
            loss_ge.backward()
            optimizer_fe.step()
            optimizer_g.step()
            total_reconstruction_loss += loss_ge.item()

            # Location Predictorの事前学習 (L_R^s) 
            optimizer_lp.zero_grad()
            
            # Source DomainのZをLocation Predictorに入力
            z_s_lp = model.feature_extractor(source_rss) # Feature Extractorは更新済み
            predicted_loc_avg, predicted_loc_r1, predicted_loc_r2, predicted_loc_r3 = model.location_predictor(z_s_lp)
            
            # L_R^s = ||y_s_avg - y_s|| + 1/3 * sum(||y_s_i - y_s||) 
            loss_rs = location_criterion(predicted_loc_avg, source_loc) + \
                      (location_criterion(predicted_loc_r1, source_loc) + \
                       location_criterion(predicted_loc_r2, source_loc) + \
                       location_criterion(predicted_loc_r3, source_loc)) / 3
            
            loss_rs.backward()
            optimizer_lp.step()
            total_source_lp_loss += loss_rs.item()

        print(f"Pre-Epoch {epoch+1}/{pretrain_epochs}, Rec Loss: {total_reconstruction_loss / len(source_train_loader):.4f}, Source LP Loss: {total_source_lp_loss / len(source_train_loader):.4f}")

    print("--- Joint Training Phase ---")
    # 共同学習フェーズ 
    for epoch in range(num_epochs):
        model.train()
        total_loss_d = 0
        total_loss_g_adv = 0
        total_loss_g_rec = 0
        total_loss_lp = 0
        total_loss_cc_f = 0
        total_loss_cc_p = 0

        # Pseudo-label metrics for current epoch
        epoch_pseudo_label_counts = {0: 0, 1: 0, 2: 0}
        epoch_pseudo_label_consistency_errors = []
        # 追加: 現在のエポックでの擬似ラベルと真のラベルの誤差リスト
        epoch_pseudo_label_true_errors = [] 

        # 追加: 現在のエポックで生成された擬似ラベルの位置情報を記憶する辞書
        current_epoch_pseudo_label_locations = {} 
        # 追加: 現在のエポックの座標ごとの統計情報を一時的に保持する辞書
        current_epoch_loc_stats = {} 

        # 現在のエポックの擬似ラベルとRSLデータを収集するリスト
        current_epoch_pseudo_data = []

        # データローダーをイテレート
        # target_train_loader は unlabeled data (x_t) 
        # source_train_loader は labeled data (x_s, y_s) 
        # 各バッチでsourceとtargetのフィンガープリントをペアリング
        # min_batches = min(len(source_train_loader), len(target_train_loader)) # Original idea for paired batches
        
        # Adjusting iteration for potentially unequal loader lengths
        # Using a cyclic iterator for the smaller dataset to ensure all samples are processed
        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)

        num_batches = max(len(source_train_loader), len(target_train_loader))

        for i in range(num_batches):
            try:
                source_rss, source_loc = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train_loader)
                source_rss, source_loc = next(source_iter)

            try:
                # target_rss, _ = next(target_iter) # Target data is unlabeled
                target_rss, target_loc = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                # target_rss, _ = next(target_iter)
                target_rss, target_loc = next(target_iter)

            source_rss, source_loc = source_rss.to(device), source_loc.to(device)
            # target_rss = target_rss.to(device)
            target_rss, target_loc = target_rss.to(device), target_loc.to(device)

            # real_labels = torch.ones(source_rss.size(0), 1).to(device)
            # fake_labels = torch.zeros(source_rss.size(0), 1).to(device)
            # そのバッチで処理される最大バッチサイズを決定
            current_max_batch_size = max(source_rss.size(0), target_rss.size(0))

            # real_labels と fake_labels を現在のバッチサイズに合わせて作成
            real_labels = torch.ones(current_max_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_max_batch_size, 1).to(device)
            
            # Domain labels
            source_domain_label = torch.zeros(source_rss.size(0), 1).to(device) # d_s = 0
            target_domain_label = torch.ones(target_rss.size(0), 1).to(device)   # d_t = 1


            # --- 1. Discriminatorの更新 (L_D) --- 
            optimizer_d.zero_grad()

            # Source Discriminator (D_s)
            # Real source data
            d_s_real, _ = model.discriminator(source_rss)
            loss_ds_real = criterion_discriminator(d_s_real, real_labels[:d_s_real.size(0)])

            # Fake source data (transformed from target to source)
            # Z from target data
            # z_t = model.feature_extractor(target_rss)
            # fake_source_rss = model.generator(z_t, source_domain_label[:z_t.size(0)])
            # d_s_fake, _ = model.discriminator(fake_source_rss.detach()) # Detach to prevent G from updating
            # loss_ds_fake = criterion_discriminator(d_s_fake, fake_labels[:d_s_fake.size(0)])

            # Fake source data (transformed from target to source)
            z_t = model.feature_extractor(target_rss)
            # source_domain_label のサイズを z_t のバッチサイズに合わせる
            current_batch_source_domain_label = torch.zeros(z_t.size(0), 1).to(device)
            fake_source_rss = model.generator(z_t, current_batch_source_domain_label) # 修正箇所1
            d_s_fake, _ = model.discriminator(fake_source_rss.detach())
            loss_ds_fake = criterion_discriminator(d_s_fake, fake_labels[:d_s_fake.size(0)])
            
            loss_ds = 0.5 * (loss_ds_real + loss_ds_fake) # Eq 5 (L_Ds) 


            # Target Discriminator (D_t)
            # Real target data
            _, d_t_real = model.discriminator(target_rss)
            loss_dt_real = criterion_discriminator(d_t_real, real_labels[:d_t_real.size(0)])

            # Fake target data (transformed from source to target)
            # Z from source data
            # z_s = model.feature_extractor(source_rss)
            # fake_target_rss = model.generator(z_s, target_domain_label[:z_s.size(0)])
            # _, d_t_fake = model.discriminator(fake_target_rss.detach()) # Detach to prevent G from updating
            # loss_dt_fake = criterion_discriminator(d_t_fake, fake_labels[:d_t_fake.size(0)])

            # Fake target data (transformed from source to target)
            z_s = model.feature_extractor(source_rss)
            # target_domain_label のサイズを z_s のバッチサイズに合わせる
            current_batch_target_domain_label = torch.ones(z_s.size(0), 1).to(device)
            fake_target_rss = model.generator(z_s, current_batch_target_domain_label) # 修正箇所2
            _, d_t_fake = model.discriminator(fake_target_rss.detach())
            loss_dt_fake = criterion_discriminator(d_t_fake, fake_labels[:d_t_fake.size(0)])

            loss_dt = 0.5 * (loss_dt_real + loss_dt_fake) # Eq 6 (L_Dt) 

            loss_d = loss_ds + loss_dt # Eq 7 (L_D) 
            
            loss_d.backward()
            optimizer_d.step()
            total_loss_d += loss_d.item()


            # --- 2. Feature Extractor, Generator, Location Predictorの更新 ---
            # これらは共同で Discriminator を騙す (敵対的学習) 
            # Generator は再構築損失 (L_GE) と敵対的損失 (Generatorの目標) 
            # Feature Extractor は敵対的損失と位置予測損失 (L_R) 
            # Location Predictor は位置予測損失 (L_R) 
            # L_CC (サイクル一貫性損失) 

            optimizer_fe.zero_grad()
            optimizer_g.zero_grad()
            optimizer_lp.zero_grad()

            # L_GE: Reconstruction Loss 
            # Source -> Z_s -> G(Z_s, d_s) -> Reconstructed Source
            z_s = model.feature_extractor(source_rss)
            reconstructed_source_rss = model.generator(z_s, source_domain_label)
            loss_ge_s = reconstruction_criterion(reconstructed_source_rss, source_rss)
            
            # Target -> Z_t -> G(Z_t, d_t) -> Reconstructed Target
            z_t = model.feature_extractor(target_rss)
            reconstructed_target_rss = model.generator(z_t, target_domain_label)
            loss_ge_t = reconstruction_criterion(reconstructed_target_rss, target_rss)
            
            loss_ge = loss_ge_s + loss_ge_t # Eq 3 (L_GE) 

            # Generatorの敵対的損失 (Discriminatorを騙す) 
            # L_GAN (G) = 0.5 * (D_s(G(Z_t, d_s)) - 1)^2 + 0.5 * (D_t(G(Z_s, d_t)) - 1)^2
            
            # Fake Source (Target -> Source)
            # z_t_for_g = model.feature_extractor(target_rss)
            # fake_source_rss_for_g = model.generator(z_t_for_g, source_domain_label[:z_t_for_g.size(0)])
            # d_s_fake_for_g, _ = model.discriminator(fake_source_rss_for_g)
            # loss_g_adv_s = criterion_generator_adv(d_s_fake_for_g, real_labels[:d_s_fake_for_g.size(0)])

            # Fake Source (Target -> Source)
            z_t_for_g = model.feature_extractor(target_rss)
            # source_domain_label のサイズを z_t_for_g のバッチサイズに合わせる
            current_batch_source_domain_label_for_g = torch.zeros(z_t_for_g.size(0), 1).to(device)
            fake_source_rss_for_g = model.generator(z_t_for_g, current_batch_source_domain_label_for_g) # 修正箇所3
            d_s_fake_for_g, _ = model.discriminator(fake_source_rss_for_g)
            loss_g_adv_s = criterion_generator_adv(d_s_fake_for_g, real_labels[:d_s_fake_for_g.size(0)])

            # Fake Target (Source -> Target)
            # z_s_for_g = model.feature_extractor(source_rss)
            # fake_target_rss_for_g = model.generator(z_s_for_g, target_domain_label[:z_s_for_g.size(0)])
            # _, d_t_fake_for_g = model.discriminator(fake_target_rss_for_g)
            # loss_g_adv_t = criterion_generator_adv(d_t_fake_for_g, real_labels[:d_t_fake_for_g.size(0)])

            # Fake Target (Source -> Target)
            z_s_for_g = model.feature_extractor(source_rss)
            # target_domain_label のサイズを z_s_for_g のバッチサイズに合わせる
            current_batch_target_domain_label_for_g = torch.ones(z_s_for_g.size(0), 1).to(device)
            fake_target_rss_for_g = model.generator(z_s_for_g, current_batch_target_domain_label_for_g) # 修正箇所4
            _, d_t_fake_for_g = model.discriminator(fake_target_rss_for_g)
            loss_g_adv_t = criterion_generator_adv(d_t_fake_for_g, real_labels[:d_t_fake_for_g.size(0)])

            loss_g_adv = 0.5 * (loss_g_adv_s + loss_g_adv_t)
            total_loss_g_adv += loss_g_adv.item()

            # L_R: Location Predictor Loss 
            # Source Domain Fingerprints (L_R^s) 
            z_s_lp = model.feature_extractor(source_rss)
            predicted_loc_avg_s, predicted_loc_r1_s, predicted_loc_r2_s, predicted_loc_r3_s = model.location_predictor(z_s_lp)
            
            loss_rs = location_criterion(predicted_loc_avg_s, source_loc) + \
                      (location_criterion(predicted_loc_r1_s, source_loc) + \
                       location_criterion(predicted_loc_r2_s, source_loc) + \
                       location_criterion(predicted_loc_r3_s, source_loc)) / 3 # Eq 10 

            # Target Domain Fingerprints (L_R^t) - Pseudo-labeling 
            # This part is simplified from Tri-net's full pseudo-labeling.
            # For a full implementation, you'd collect pseudo-labels over epochs,
            # refine them, and add to the loss. Here, a simpler approach.
            # Tri-net details for pseudo-labeling:  Same Prediction, Confident Prediction, Stable Prediction.
            # Using current batch to simulate pseudo-labeling.

            pseudo_labeled_target_locs = []
            valid_pseudo_labels_count = 0
            
            # Dropout for stable prediction 
            model.location_predictor.train() # Enable dropout for stable prediction check
            
            z_t_lp_raw = model.feature_extractor(target_rss)

            # Get multiple predictions with dropout enabled
            num_dropout_predictions = 5 # Q times for stability 
            r1_preds_dropout = [model.location_predictor(z_t_lp_raw)[1] for _ in range(num_dropout_predictions)]
            r2_preds_dropout = [model.location_predictor(z_t_lp_raw)[2] for _ in range(num_dropout_predictions)]
            r3_preds_dropout = [model.location_predictor(z_t_lp_raw)[3] for _ in range(num_dropout_predictions)]

            # Disable dropout for confident/same prediction (using direct predictions)
            model.location_predictor.eval() # Disable dropout for regular inference
            with torch.no_grad(): # No gradient calculation for this part
                _, r1_pred_no_dropout, r2_pred_no_dropout, r3_pred_no_dropout = model.location_predictor(z_t_lp_raw)

            # Re-enable dropout for next training iteration
            model.location_predictor.train()

            for idx in range(target_rss.size(0)):
                # Same Prediction 
                # Compare two of the three sub-modules (e.g., R2 and R3 for R1's pseudo-label)
                # Note: epsilon (ε) needs to be defined based on normalized location range.
                epsilon_loc = epsilon_tri_net # A small quantity for distance comparison 

                # ターゲット座標の真の値を元のスケールに戻す
                # Tupleに変換して辞書のキーとして使用
                original_target_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(target_loc[idx].cpu().numpy().reshape(1, -1)).flatten()
                original_target_loc_tuple = tuple(original_target_loc_unscaled)

                # current_epoch_loc_stats の初期化（もしその座標が初めて出現したら）
                if original_target_loc_tuple not in current_epoch_loc_stats:
                    current_epoch_loc_stats[original_target_loc_tuple] = {'generated_count': 0, 'total_true_error': 0.0}

                # R1's pseudo-label is from R2, R3
                if torch.norm(r2_pred_no_dropout[idx] - r3_pred_no_dropout[idx]) < epsilon_loc:
                    # Confident Prediction (R1 different from R2, R3) 
                    if torch.norm(r1_pred_no_dropout[idx] - r2_pred_no_dropout[idx]) > epsilon_loc and \
                       torch.norm(r1_pred_no_dropout[idx] - r3_pred_no_dropout[idx]) > epsilon_loc:
                        # Stable Prediction 
                        is_stable = True
                        for q_idx in range(num_dropout_predictions):
                            if torch.norm(r2_preds_dropout[q_idx][idx] - r2_pred_no_dropout[idx]) >= epsilon_loc or \
                               torch.norm(r3_preds_dropout[q_idx][idx] - r3_pred_no_dropout[idx]) >= epsilon_loc:
                                is_stable = False
                                break
                        if is_stable:
                            # print("R1: add pseudo")
                            pseudo_label = (r2_pred_no_dropout[idx] + r3_pred_no_dropout[idx]) / 2 # Eq 11 
                            pseudo_labeled_target_locs.append((z_t_lp_raw[idx], pseudo_label, 0)) # Store (z, pseudo_y, sub_module_idx for R1)
                            valid_pseudo_labels_count += 1

                            epoch_pseudo_label_counts[0] += 1
                            consistency_error = torch.norm(pseudo_label - r2_pred_no_dropout[idx]) + \
                                                torch.norm(pseudo_label - r3_pred_no_dropout[idx])
                            epoch_pseudo_label_consistency_errors.append(consistency_error.item())
                            
                            # 追加: 擬似ラベルと真のラベルの誤差を計算 (アンノーマライズしてから)
                            pseudo_label_unscaled = data_scalers['loc_scaler'].inverse_transform(pseudo_label.cpu().numpy().reshape(1, -1)).flatten()
                            true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(target_loc[idx].cpu().numpy().reshape(1, -1)).flatten()
                            true_error = np.sqrt(np.sum((pseudo_label_unscaled - true_loc_unscaled)**2))
                            epoch_pseudo_label_true_errors.append(true_error) # この行を追加

                            # 追加: 擬似ラベルの位置を元のターゲット座標に紐付けて記憶
                            if original_target_loc_tuple not in current_epoch_pseudo_label_locations:
                                current_epoch_pseudo_label_locations[original_target_loc_tuple] = []
                            current_epoch_pseudo_label_locations[original_target_loc_tuple].append(pseudo_label_unscaled.tolist())

                             # 追加: 座標ごとの統計を更新
                            current_epoch_loc_stats[original_target_loc_tuple]['generated_count'] += 1
                            current_epoch_loc_stats[original_target_loc_tuple]['total_true_error'] += true_error

                            # --- 修正/追加: RSL特徴量と擬似ラベルを収集 ---
                            # target_rss[idx] は RSL 特徴
                            # pseudo_label は正規化された擬似ラベル位置
                            current_epoch_pseudo_data.append((target_rss[idx].cpu(), pseudo_label.cpu()))

                # R2's pseudo-label is from R1, R3
                if torch.norm(r1_pred_no_dropout[idx] - r3_pred_no_dropout[idx]) < epsilon_loc:
                     if torch.norm(r2_pred_no_dropout[idx] - r1_pred_no_dropout[idx]) > epsilon_loc and \
                        torch.norm(r2_pred_no_dropout[idx] - r3_pred_no_dropout[idx]) > epsilon_loc:
                        is_stable = True
                        for q_idx in range(num_dropout_predictions):
                            if torch.norm(r1_preds_dropout[q_idx][idx] - r1_pred_no_dropout[idx]) >= epsilon_loc or \
                               torch.norm(r3_preds_dropout[q_idx][idx] - r3_pred_no_dropout[idx]) >= epsilon_loc:
                                is_stable = False
                                break
                        if is_stable:
                            # print("R2: add pseudo")
                            pseudo_label = (r1_pred_no_dropout[idx] + r3_pred_no_dropout[idx]) / 2
                            pseudo_labeled_target_locs.append((z_t_lp_raw[idx], pseudo_label, 1)) # sub_module_idx for R2
                            valid_pseudo_labels_count += 1

                            epoch_pseudo_label_counts[1] += 1
                            consistency_error = torch.norm(pseudo_label - r1_pred_no_dropout[idx]) + \
                                                torch.norm(pseudo_label - r3_pred_no_dropout[idx])
                            epoch_pseudo_label_consistency_errors.append(consistency_error.item())
                            
                            # 追加: 擬似ラベルと真のラベルの誤差を計算
                            pseudo_label_unscaled = data_scalers['loc_scaler'].inverse_transform(pseudo_label.cpu().numpy().reshape(1, -1)).flatten()
                            true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(target_loc[idx].cpu().numpy().reshape(1, -1)).flatten()
                            true_error = np.sqrt(np.sum((pseudo_label_unscaled - true_loc_unscaled)**2))
                            epoch_pseudo_label_true_errors.append(true_error) # この行を追加

                            # 追加: 擬似ラベルの位置を元のターゲット座標に紐付けて記憶
                            if original_target_loc_tuple not in current_epoch_pseudo_label_locations:
                                current_epoch_pseudo_label_locations[original_target_loc_tuple] = []
                            current_epoch_pseudo_label_locations[original_target_loc_tuple].append(pseudo_label_unscaled.tolist())

                             # 追加: 座標ごとの統計を更新
                            current_epoch_loc_stats[original_target_loc_tuple]['generated_count'] += 1
                            current_epoch_loc_stats[original_target_loc_tuple]['total_true_error'] += true_error

                            # --- 修正/追加: RSL特徴量と擬似ラベルを収集 ---
                            current_epoch_pseudo_data.append((target_rss[idx].cpu(), pseudo_label.cpu()))

                # R3's pseudo-label is from R1, R2
                if torch.norm(r1_pred_no_dropout[idx] - r2_pred_no_dropout[idx]) < epsilon_loc:
                     if torch.norm(r3_pred_no_dropout[idx] - r1_pred_no_dropout[idx]) > epsilon_loc and \
                        torch.norm(r3_pred_no_dropout[idx] - r2_pred_no_dropout[idx]) > epsilon_loc:
                        is_stable = True
                        for q_idx in range(num_dropout_predictions):
                            if torch.norm(r1_preds_dropout[q_idx][idx] - r1_pred_no_dropout[idx]) >= epsilon_loc or \
                               torch.norm(r2_preds_dropout[q_idx][idx] - r2_pred_no_dropout[idx]) >= epsilon_loc:
                                is_stable = False
                                break
                        if is_stable:
                            # print("R3: add pseudo")
                            pseudo_label = (r1_pred_no_dropout[idx] + r2_pred_no_dropout[idx]) / 2
                            pseudo_labeled_target_locs.append((z_t_lp_raw[idx], pseudo_label, 2)) # sub_module_idx for R3
                            valid_pseudo_labels_count += 1

                            epoch_pseudo_label_counts[2] += 1
                            consistency_error = torch.norm(pseudo_label - r1_pred_no_dropout[idx]) + \
                                                torch.norm(pseudo_label - r2_pred_no_dropout[idx])
                            epoch_pseudo_label_consistency_errors.append(consistency_error.item())
                            
                            # 追加: 擬似ラベルと真のラベルの誤差を計算
                            pseudo_label_unscaled = data_scalers['loc_scaler'].inverse_transform(pseudo_label.cpu().numpy().reshape(1, -1)).flatten()
                            true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(target_loc[idx].cpu().numpy().reshape(1, -1)).flatten()
                            true_error = np.sqrt(np.sum((pseudo_label_unscaled - true_loc_unscaled)**2))
                            epoch_pseudo_label_true_errors.append(true_error) # この行を追加

                            # 追加: 擬似ラベルの位置を元のターゲット座標に紐付けて記憶
                            if original_target_loc_tuple not in current_epoch_pseudo_label_locations:
                                current_epoch_pseudo_label_locations[original_target_loc_tuple] = []
                            current_epoch_pseudo_label_locations[original_target_loc_tuple].append(pseudo_label_unscaled.tolist())

                             # 追加: 座標ごとの統計を更新
                            current_epoch_loc_stats[original_target_loc_tuple]['generated_count'] += 1
                            current_epoch_loc_stats[original_target_loc_tuple]['total_true_error'] += true_error

                            # --- 修正/追加: RSL特徴量と擬似ラベルを収集 ---
                            current_epoch_pseudo_data.append((target_rss[idx].cpu(), pseudo_label.cpu()))

            loss_rt = 0
            if valid_pseudo_labels_count > 0:
                # Calculate L_R^t using collected pseudo-labels
                # This needs to be done on the specific sub-module for which the pseudo-label was generated
                # And z_pl needs to be passed through the location_predictor again
                for z_pl, y_pl, submodule_idx in pseudo_labeled_target_locs:
                    z_pl = z_pl.unsqueeze(0) # Add batch dimension
                    _, r1_p, r2_p, r3_p = model.location_predictor(z_pl)
                    if submodule_idx == 0: # R1's pseudo-label
                        loss_rt += location_criterion(r1_p, y_pl.unsqueeze(0))
                    elif submodule_idx == 1: # R2's pseudo-label
                        loss_rt += location_criterion(r2_p, y_pl.unsqueeze(0))
                    elif submodule_idx == 2: # R3's pseudo-label
                        loss_rt += location_criterion(r3_p, y_pl.unsqueeze(0))
                loss_rt /= valid_pseudo_labels_count

            loss_r = loss_rs + loss_rt # Eq 13 (L_R) 
            total_loss_lp += loss_r.item()


            # L_CC: Cycle Consistency Loss 
            # L_CC^f: Feature-level cycle consistency 
            # Source -> G(d_t) -> Fake_Target -> E -> Z_st
            z_st = model.feature_extractor(fake_target_rss_for_g) # fake_target_rss_for_g from G(Z_s, d_t)
            loss_cc_f = cycle_consistency_criterion_f(z_st, z_s_for_g) # Z_s from source_rss 
            
            # L_CC^p: Prediction-level cycle consistency 
            # Fake_Target (from Source) -> Location Predictor -> y_st (should be close to y_s)
            predicted_loc_avg_st, predicted_loc_r1_st, predicted_loc_r2_st, predicted_loc_r3_st = model.location_predictor(z_st)
            
            # ||y_st_avg - y_s|| + 1/3 * sum(||y_st_i - y_s||) 
            loss_cc_p = cycle_consistency_criterion_p(predicted_loc_avg_st, source_loc) + \
                        (cycle_consistency_criterion_p(predicted_loc_r1_st, source_loc) + \
                         cycle_consistency_criterion_p(predicted_loc_r2_st, source_loc) + \
                         cycle_consistency_criterion_p(predicted_loc_r3_st, source_loc)) / 3 # Eq 16 

            loss_cc = loss_cc_f + loss_cc_p # Eq 17 (L_CC) 
            total_loss_cc_f += loss_cc_f.item()
            total_loss_cc_p += loss_cc_p.item()


            # Total Objective (L) 
            # λ_D, λ_R, λ_CC はハイパーパラメータ 
            # Generator は敵対者なので、G/FE/LP の更新では Discriminator の勾配は反転して伝播させるイメージ
            # PyTorchではそれぞれの損失を合計して逆伝播させることで実現
            loss_total = loss_ge + lambda_D * loss_g_adv + lambda_R * loss_r + lambda_CC * loss_cc # Eq 18 (L) 
            
            loss_total.backward() # Backward on combined loss
            optimizer_fe.step()
            optimizer_g.step()
            optimizer_lp.step()
        
        # Record epoch losses (変更: 各エポックの損失を履歴リストに追加)
        total_loss_d_history.append(total_loss_d / num_batches)
        total_loss_g_adv_history.append(total_loss_g_adv / num_batches)
        total_loss_g_rec_history.append(total_loss_g_rec / num_batches)
        total_loss_lp_history.append(total_loss_lp / num_batches)
        total_loss_cc_f_history.append(total_loss_cc_f / num_batches)
        total_loss_cc_p_history.append(total_loss_cc_p / num_batches)

        # エポックごとの進捗表示
        # print(f"Epoch {epoch+1}/{num_epochs}, D_Loss: {total_loss_d / num_batches:.4f}, G_Adv_Loss: {total_loss_g_adv / num_batches:.4f}, "
        #       f"LP_Loss: {total_loss_lp / num_batches:.4f}, CC_F_Loss: {total_loss_cc_f / num_batches:.4f}, CC_P_Loss: {total_loss_cc_p / num_batches:.4f}")

        # エポック終了時、履歴リストに追加（for i in range(num_batches): ループの直後）
        total_pseudo_labels = sum(epoch_pseudo_label_counts.values())
        avg_pseudo_label_consistency_error = sum(epoch_pseudo_label_consistency_errors) / len(epoch_pseudo_label_consistency_errors) if len(epoch_pseudo_label_consistency_errors) > 0 else 0
        # 追加: 擬似ラベルと真のラベルの平均誤差を計算
        avg_pseudo_label_true_error = sum(epoch_pseudo_label_true_errors) / len(epoch_pseudo_label_true_errors) if len(epoch_pseudo_label_true_errors) > 0 else 0 

        pseudo_label_counts_history.append(total_pseudo_labels)
        pseudo_label_consistency_error_history.append(avg_pseudo_label_consistency_error)
        # 追加: 履歴リストに平均誤差を追加
        pseudo_label_true_error_history.append(avg_pseudo_label_true_error) 

        # 追加: 現在のエポックの擬似ラベル位置情報を全体のリストに追加
        all_epochs_pseudo_label_locations.append(current_epoch_pseudo_label_locations)

        # 追加: current_epoch_loc_stats を最終的な形式に変換して all_epochs_detailed_pseudo_label_stats に追加
        final_epoch_loc_stats = {}
        for loc_tuple, stats in current_epoch_loc_stats.items():
            total_data_for_loc = target_train_total_counts.get(loc_tuple, 1) # 0除算を防ぐ
            generated_count = stats['generated_count']
            ratio = generated_count / total_data_for_loc if total_data_for_loc > 0 else 0
            avg_true_error = stats['total_true_error'] / generated_count if generated_count > 0 else 0.0
            
            final_epoch_loc_stats[loc_tuple] = {
                'total_data_count': total_data_for_loc,
                'generated_count': generated_count,
                'ratio': ratio,
                'avg_true_error': avg_true_error
            }
        all_epochs_detailed_pseudo_label_stats.append(final_epoch_loc_stats)

        print(f"Epoch {epoch+1}/{num_epochs}, D_Loss: {total_loss_d / num_batches:.4f}, G_Adv_Loss: {total_loss_g_adv / num_batches:.4f}, "
              f"LP_Loss: {total_loss_lp / num_batches:.4f}, CC_F_Loss: {total_loss_cc_f / num_batches:.4f}, CC_P_Loss: {total_loss_cc_p / num_batches:.4f}, "
              f"Pseudo-Labels: {total_pseudo_labels}, Pseudo-Label Consistency Error: {avg_pseudo_label_consistency_error:.4f}, "
              f"Pseudo-Label True Error: {avg_pseudo_label_true_error:.4f}") # print文を修正

        # テストセットでの評価 (測位精度)
        if (epoch + 1) % 10 == 0: # 10エポックごとに評価
            model.eval()
            total_test_distance = 0
            with torch.no_grad():
                for test_rss, test_loc in target_test_loader:
                    test_rss, test_loc = test_rss.to(device), test_loc.to(device)
                    z_test = model.feature_extractor(test_rss)
                    predicted_loc_avg, _, _, _ = model.location_predictor(z_test)
                    
                    # 予測位置を元のスケールに戻す
                    predicted_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(predicted_loc_avg.cpu().numpy())
                    true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(test_loc.cpu().numpy())

                    # ユークリッド距離で誤差を計算 (m単位) 
                    distances = np.sqrt(np.sum((predicted_loc_unscaled - true_loc_unscaled)**2, axis=1))
                    total_test_distance += np.sum(distances)
            
            avg_test_distance = total_test_distance / len(target_test_loader.dataset)
            # テスト誤差と対応するエポックを記録 (追加)
            test_error_history.append(avg_test_distance)
            test_error_epochs_recorded.append(epoch + 1)
            print(f"--- Epoch {epoch+1} Test Localization Error: {avg_test_distance:.4f} m --- ")
            model.train() # Set back to train mode

        # エポックの終わりに、現在のエポックの擬似ラベルデータを保存
        if epoch == num_epochs - 1:
            last_epoch_pseudo_data = current_epoch_pseudo_data
            print(f"Collected {len(last_epoch_pseudo_data)} pseudo-labeled samples from the last epoch.")

    print("Training finished.")

    # # 訓練履歴を返す (変更: 戻り値を追加)
    # return (total_loss_d_history, total_loss_g_adv_history, total_loss_g_rec_history,
    #         total_loss_lp_history, total_loss_cc_f_history, total_loss_cc_p_history,
    #         test_error_history, test_error_epochs_recorded)

    # train_transloc 関数の return 文を修正
    return (total_loss_d_history, total_loss_g_adv_history, total_loss_g_rec_history,
            total_loss_lp_history, total_loss_cc_f_history, total_loss_cc_p_history,
            test_error_history, test_error_epochs_recorded,
            pseudo_label_counts_history, pseudo_label_consistency_error_history,
            pseudo_label_true_error_history, # この行を修正
            all_epochs_pseudo_label_locations, # この行を修正
            all_epochs_detailed_pseudo_label_stats,
            last_epoch_pseudo_data) # この行を追加


# --- 実行部分 ---
if __name__ == "__main__":
    # データローダーの作成
    # place_name = 'OfficeP2'
    # train_fnum = '1'
    # test_fnum = '3'
    # source_train_path = f'./data/{place_name}/csv/{place_name}_{train_fnum}_training.csv'
    # target_train_path = f'./data/{place_name}/csv/{place_name}_{test_fnum}_training.csv'
    # target_test_path = f'./data/{place_name}/csv/{place_name}_{test_fnum}_testing.csv'

    # date = '20251030'
    train_date = '20251228'
    test_date = '20251228'
    train_scene = '12Anchors_1Tag_non_obst'#'non_obst'#'half_wall_A'#'Tripod_non_obst'
    test_scene = '12Anchors_1Tag_wallA_15'#'Aluminum_foilW_A_35'#'1805NLOS_Aluminu_foilW'#'Tripod_aluminum_foil_whiteboard_A'#'wall_A'#'wall_A'#'non_obst'#'1_lounges_whiteboard_A'#'wall'
    # source_train_path = f'./data/uwb/processed_uwb_full_features_data_{train_scene}.csv'
    # target_train_path = f'./data/uwb/processed_uwb_full_features_data_{test_scene}_train_split.csv'
    # target_test_path = f'./data/uwb/processed_uwb_full_features_data_{test_scene}_test_split.csv'
    # source_train_path = f'./data/uwb/{date}/processed_uwb_full_features_data_{train_scene}_train_split.csv'
    # target_train_path = f'./data/uwb/{date}/processed_uwb_full_features_data_{test_scene}_train_split.csv'
    # target_test_path = f'./data/uwb/{date}/processed_uwb_full_features_data_{test_scene}_test_split.csv'
    source_train_path = f'./data/uwb/{train_date}/processed_uwb_full_features_data_{train_scene}_train_split.csv'
    target_train_path = f'./data/uwb/{test_date}/processed_uwb_full_features_data_{test_scene}_train_split.csv'
    target_test_path = f'./data/uwb/{test_date}/processed_uwb_full_features_data_{test_scene}_test_split.csv'

    # ap_filter_list = [
    #     # 'AP910',
    #     'AP4250',
    #     'AP4245',
    #     'AP17057',
    #     # 'AP23196'
    # ]
    ap_filter_list = { # 12Anchors
        # 'AP4524': 4524, # (0, -1.5)
        # 'AP5307': 5307, # (0, 3)
        'AP36794': 36794, # (-1.5, 0)
        'AP37248': 37248, # (-1.5, 1.5)
        # 'AP37051': 37051, # (3, 1.5)
        # 'AP7091': 7091, # (3, 0)
        'AP1805': 1805, # (1.5, -1.5)
        'AP910': 910, # (3, 3)
        'AP4250': 4250, # (0.75, 3)
        'AP17057': 17057, # (0.75, -1.5)
        # 'AP37045': 37045, # (-1.5, 0.75)
        # 'AP23196': 23196 # (3, 0.75)
    }

    source_train_loader, target_train_loader, target_test_loader, data_scalers = create_dataloaders(
        source_train_path, target_train_path, target_test_path, ap_filter_list
    )

    # 入力/出力次元の取得
    sample_rss, _ = next(iter(source_train_loader))
    input_dim = sample_rss.shape[1] # RSS特徴の次元
    output_loc_dim = 2 # X, Y座標 

    print(f"Input Dimension (RSS features): {input_dim}")
    print(f"Output Dimension (Location features): {output_loc_dim}")
    print(f"Number of samples in source_train: {len(source_train_loader.dataset)}")
    print(f"Number of samples in target_train: {len(target_train_loader.dataset)}")
    print(f"Number of samples in target_test: {len(target_test_loader.dataset)}")

    # TransLocモデルの初期化
    z_dim = 16 # ドメイン不変特徴の次元 
    domain_dim = 1 # ドメインラベルの次元 (バイナリ: Source=0, Target=1)
    
    model = TransLoc(input_dim, z_dim, output_loc_dim, domain_dim).to(device)
    print("TransLoc model initialized.")
    print(model) # モデル構造の確認

    # 学習の実行
    # ハイパーパラメータのデフォルト値は論文の「Hyper-parameter Selection」セクションを参照 
    # lambda_D, lambda_R, lambda_CC のデフォルト値は1 
    # η (eta_gradient_reversal) のデフォルト値は10 
    # train_transloc(model, source_train_loader, target_train_loader, target_test_loader, data_scalers, # model評価プロット前
    # loss_d_hist, loss_g_adv_hist, loss_g_rec_hist, loss_lp_hist, \
    # loss_cc_f_hist, loss_cc_p_hist, test_err_hist, test_err_epochs = train_transloc(
    #     model, source_train_loader, target_train_loader, target_test_loader, data_scalers,
    #                num_epochs=50,#200, # 論文の実験期間は3ヶ月 (長期間) 
    #                lr_fe_lp=0.0002, # Adamの学習率は論文のImageNet実験から参考に (DANN: 0.0002)
    #                lr_g=0.0002,
    #                lr_d=0.0002,
    #                lambda_D=1, lambda_R=1, lambda_CC=1,
    #                epsilon_tri_net=1e-4) # Tri-netのepsilon (非常に小さい量) 
    loss_d_hist, loss_g_adv_hist, loss_g_rec_hist, loss_lp_hist, \
    loss_cc_f_hist, loss_cc_p_hist, test_err_hist, test_err_epochs, \
    pseudo_label_counts_hist, pseudo_label_consistency_err_hist, \
    pseudo_label_true_error_hist, \
    all_epochs_pseudo_label_locations, \
    all_epochs_detailed_pseudo_label_stats, \
    last_epoch_pseudo_data = train_transloc( # この行を修正
        model, source_train_loader, target_train_loader, target_test_loader, data_scalers,
                    num_epochs=70, #200,
                    lr_fe_lp=0.0002,
                    lr_g=0.0002,
                    lr_d=0.0002,
                    lambda_D=1, lambda_R=1, lambda_CC=1,
                    epsilon_tri_net=1e-2)

    # # モデルの保存 (オプション)
    torch.save(model.state_dict(), f"./output/transloc_model_train_{train_scene}test_{test_scene}.pth")
    print("Model saved to transloc_model.pth")

    # # テストデータで最終評価
    model.eval()
    total_test_distance = 0
    all_true_locs = []
    all_predicted_locs = []
    with torch.no_grad():
        for test_rss, test_loc in target_test_loader:
            test_rss, test_loc = test_rss.to(device), test_loc.to(device)
            z_test = model.feature_extractor(test_rss)
            predicted_loc_avg, _, _, _ = model.location_predictor(z_test)
            
            predicted_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(predicted_loc_avg.cpu().numpy())
            true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(test_loc.cpu().numpy())

            distances = np.sqrt(np.sum((predicted_loc_unscaled - true_loc_unscaled)**2, axis=1))
            total_test_distance += np.sum(distances)

            all_true_locs.extend(true_loc_unscaled)
            all_predicted_locs.extend(predicted_loc_unscaled)
    
    avg_test_distance = total_test_distance / len(target_test_loader.dataset)
    # print(f"\nFinal Test Localization Error: {avg_test_distance:.4f} m ")
    print(f"\nFinal Test Localization Error: {avg_test_distance} m ")
    with open(f"./output/localization_error_test_rsl.txt", "a", encoding="utf-8") as f:
        # print(f"\ntrain_{train_scene}_test_{test_scene}\nFinal Test Localization Error: {avg_test_distance:.4f} m ", file=f)
        # print(f"\ntrain_{train_scene}_test_{test_scene}\nFinal Test Localization Error: {avg_test_distance} m ", file=f)
        print(f"\ntrain_{train_scene}_{train_date}_test_{test_scene}_{test_date}\nFinal Test Localization Error: {avg_test_distance} m ", file=f)

    # all_true_locs = np.array(all_true_locs)
    # all_predicted_locs = np.array(all_predicted_locs)

    # plt.figure(figsize=(10, 8))
    # plt.scatter(all_true_locs[:, 0], all_true_locs[:, 1], c='blue', marker='x', label='True Locations', alpha=0.7)
    # plt.scatter(all_predicted_locs[:, 0], all_predicted_locs[:, 1], c='red', marker='o', label='Predicted Locations', alpha=0.7)

    # # 真の位置と予測位置を結ぶ線を描画 (誤差ベクトルの可視化)
    # for i in range(len(all_true_locs)):
    #     plt.plot([all_true_locs[i, 0], all_predicted_locs[i, 0]], 
    #              [all_true_locs[i, 1], all_predicted_locs[i, 1]], 
    #              'k-', linewidth=0.5, alpha=0.4) # 黒い線

    # plt.xlabel('X Coordinate (m)')
    # plt.ylabel('Y Coordinate (m)')
    # plt.title(f'Localization Results: True vs. Predicted Locations (Test Scene: {test_scene})')
    # plt.legend()
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を等しくして、歪みをなくす
    # plt.tight_layout()
    # plt.savefig(f'./output/localization_results_plot_train_{train_scene}_test_{test_scene}.png')
    # plt.close()

    # print(f"Localization results plot generated: localization_results_plot_train_{train_scene}_test_{test_scene}.png")

    # --- 測位結果のプロット (平均プロット点ごとに表示) ---
    print("\nPlotting final localization results with average predicted points...")
    
    all_true_locs = np.array(all_true_locs)
    all_predicted_locs = np.array(all_predicted_locs)

    # ユニークな真の位置とその予測位置をグループ化
    # Key: tuple(true_x, true_y), Value: list of [pred_x, pred_y]
    grouped_predictions = {}
    for i in range(len(all_true_locs)):
        # NumPy配列を辞書のキーとして使うためにタプルに変換
        true_loc_tuple = tuple(all_true_locs[i]) 
        if true_loc_tuple not in grouped_predictions:
            grouped_predictions[true_loc_tuple] = []
        grouped_predictions[true_loc_tuple].append(all_predicted_locs[i])

    # 各ユニークな真の位置に対応する平均予測位置を計算
    unique_true_locs_avg = []
    avg_predicted_locs = []
    for true_loc_tuple, pred_locs_list in grouped_predictions.items():
        unique_true_locs_avg.append(list(true_loc_tuple)) # リストに戻す
        avg_pred_for_this_true_loc = np.mean(pred_locs_list, axis=0)
        avg_predicted_locs.append(avg_pred_for_this_true_loc)

    unique_true_locs_avg = np.array(unique_true_locs_avg)
    avg_predicted_locs = np.array(avg_predicted_locs)

    plt.figure(figsize=(10, 8))
    # ユニークな真の位置をプロット (大きめのマーカー)
    plt.scatter(unique_true_locs_avg[:, 0], unique_true_locs_avg[:, 1], 
                c='blue', marker='o', s=100, label='True Locations (Unique Points)', alpha=0.9)
    # 平均予測位置をプロット (大きめのマーカー)
    plt.scatter(avg_predicted_locs[:, 0], avg_predicted_locs[:, 1], 
                c='red', marker='x', s=100, label='Average Predicted Locations', alpha=0.9)

    # 各真の位置と平均予測位置を結ぶ線を描画 (誤差ベクトル)
    for i in range(len(unique_true_locs_avg)):
        plt.plot([unique_true_locs_avg[i, 0], avg_predicted_locs[i, 0]],
                 [unique_true_locs_avg[i, 1], avg_predicted_locs[i, 1]],
                 'k-', linewidth=1.0, alpha=0.6) # 黒線、太め、半透明

    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title(f'Final Localization Results: True vs. Average Predicted (Test Scene: {test_scene})')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を等しくして、歪みをなくす
    plt.tight_layout()
    # plt.savefig(f'./output/final_localization_avg_plot_train_{train_scene}_test_{test_scene}.png')
    plt.savefig(f'./output/final_localization_avg_plot_train_{train_scene}_{train_date}_test_{test_scene}_{test_date}_transloc_rsl.png')
    plt.close()
    print(f"Final localization average plot generated: final_localization_avg_plot_train_{train_scene}_test_{test_scene}.png")

    # --- プロットコードの追加 --- (追加: 訓練履歴をプロット)
    epochs_range = range(1, len(loss_d_hist) + 1)

    # Plotting Training Losses
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, loss_d_hist, label='Discriminator Loss')
    plt.plot(epochs_range, loss_g_adv_hist, label='Generator Adversarial Loss')
    plt.plot(epochs_range, loss_g_rec_hist, label='Generator Reconstruction Loss')
    plt.plot(epochs_range, loss_lp_hist, label='Location Predictor Loss')
    plt.plot(epochs_range, loss_cc_f_hist, label='Cycle Consistency Feature Loss')
    plt.plot(epochs_range, loss_cc_p_hist, label='Cycle Consistency Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('TransLoc Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'./output/transloc_training_losses_train_{train_scene}_test_{test_scene}.png')
    plt.savefig(f'./output/transloc_training_losses_train_{train_scene}_{train_date}_test_{test_scene}_{test_date}_rsl.png')
    plt.close()

    # Plotting Test Localization Error
    plt.figure(figsize=(10, 6))
    plt.plot(test_err_epochs, test_err_hist, marker='o', linestyle='-', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Localization Error (m)')
    plt.title('TransLoc Test Localization Error Over Epochs')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'./output/transloc_test_error_train_{train_scene}_test_{test_scene}.png')
    plt.savefig(f'./output/transloc_test_error_train_{train_scene}_{train_date}_test_{test_scene}_{test_date}_rsl.png')
    plt.close()

    print("Plots generated: transloc_training_losses.png and transloc_test_error.png")


    # Plotting Pseudo-Label True Error (このブロックを追加)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, pseudo_label_true_error_hist, marker='o', linestyle='-', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Average Pseudo-Label True Error (m)')
    plt.title('Average Pseudo-Label True Error Per Epoch on Target Training Data')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'./output/transloc_pseudo_label_true_error_train{train_scene}test{test_scene}.png')
    plt.savefig(f'./output/transloc_pseudo_label_true_error_train_{train_scene}_{train_date}_test_{test_scene}_{test_date}_rsl.png')
    plt.close()

    # Writing pseudo-label metrics to text file:
    test_error_map = {test_err_epochs[i]: test_err_hist[i] for i in range(len(test_err_epochs))}

    with open(f"./output/pseudo_label_metrics_rsl.txt", "a", encoding="utf-8") as f:
        f.write(f"train{train_scene}, test{test_scene}\n")
        f.write(f"Epoch,Total Pseudo-Labels,Avg Pseudo-Label Consistency Error,Avg Pseudo-Label True Error,Test Localization Error (m)\n") # ヘッダー修正
        for i in range(len(epochs_range)):
            epoch = epochs_range[i]
            pseudo_count = pseudo_label_counts_hist[i]
            consistency_err = pseudo_label_consistency_err_hist[i]
            true_err = pseudo_label_true_error_hist[i] # 新しいエラー値
            test_err = test_error_map.get(epoch)
            
            test_err_str = f"{test_err:.4f}" if test_err is not None else "N/A"
            f.write(f"{epoch},{pseudo_count},{consistency_err:.4f},{true_err:.4f},{test_err_str}\n") # 出力フォーマット修正


    print(f"Plots generated: ..., /transloc_pseudo_label_true_error_train{train_scene}test{test_scene}_rsl.png") # print文修正
    print(f"Pseudo-label metrics saved to /pseudo_label_metrics_train{train_scene}test{test_scene}_rsl.txt")

    # Plotting Pseudo-Label Counts
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, pseudo_label_counts_hist, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Pseudo-Labels Generated')
    plt.title('Total Pseudo-Labels Generated Per Epoch')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'./output/transloc_pseudo_label_counts_train{train_scene}test{test_scene}.png')
    plt.savefig(f'./output/transloc_pseudo_label_counts_train{train_scene}_{train_date}_test{test_scene}_{test_date}_rsl.png')
    plt.close()

    print(f"Pseudo-label counts plot saved to /transloc_pseudo_label_counts_train{train_scene}test{test_scene}_rsl.png")

    # 擬似ラベルの位置情報をテキストファイルに保存 (追加)
    pseudo_label_locations_filename = f"./output/pseudo_label_each_locations_rsl.txt"
    with open(pseudo_label_locations_filename, "a", encoding="utf-8") as f:
        # f.write(f"=== train {train_scene}, test {test_scene}")
        f.write(f"=== train {train_scene} {train_date}, test {test_scene} {test_date}")
        for epoch_idx, epoch_data in enumerate(all_epochs_pseudo_label_locations):
            f.write(f"--- Epoch {epoch_idx + 1} ---\n")
            for target_loc, pseudo_labels in epoch_data.items():
                f.write(f"  Target Location: ({target_loc[0]:.2f}, {target_loc[1]:.2f}), Count: {len(pseudo_labels)}\n")
                for pl in pseudo_labels:
                    f.write(f"    Pseudo-Label: ({pl[0]:.2f}, {pl[1]:.2f})\n")
            f.write("\n")

    print(f"Pseudo-label locations saved to {pseudo_label_locations_filename}")

    # 座標ごとの詳細な擬似ラベル統計をテキストファイルに保存 (追加)
    detailed_stats_filename = f"./output/detailed_pseudo_label_stats_rsl.txt"
    with open(detailed_stats_filename, "a", encoding="utf-8") as f:
        # f.write(f"\n=== train {train_scene}, test {test_scene} ===\n")
        f.write(f"\n=== train {train_scene} {train_date}, test {test_scene} {test_date} ===\n")
        for epoch_idx, epoch_stats in enumerate(all_epochs_detailed_pseudo_label_stats):
            f.write(f"--- Epoch {epoch_idx + 1} ---\n")
            # 座標でソートして出力すると見やすいかもしれません
            for target_loc_tuple in sorted(epoch_stats.keys()):
                stats = epoch_stats[target_loc_tuple]
                f.write(f"  Target Location: ({target_loc_tuple[0]:.2f}, {target_loc_tuple[1]:.2f})\n")
                f.write(f"    Total Data Count for this location: {stats['total_data_count']}\n")
                f.write(f"    Generated Pseudo-Label Count: {stats['generated_count']}\n")
                f.write(f"    Ratio of Pseudo-Labels Generated: {stats['ratio']:.4f}\n")
                f.write(f"    Avg True Error for Generated Pseudo-Labels: {stats['avg_true_error']:.4f} m\n")
            f.write("\n")

    print(f"Detailed pseudo-label statistics saved to {detailed_stats_filename}")

    # 各点ごとの誤差（ユークリッド距離）を計算
    errors = np.linalg.norm(unique_true_locs_avg - avg_predicted_locs, axis=1)

    # 各真の位置と誤差を対応づけて出力
    for i, err in enumerate(errors):
        print(f"Point {i}: True=({unique_true_locs_avg[i,0]:.2f}, {unique_true_locs_avg[i,1]:.2f}), "
            f"Pred=({avg_predicted_locs[i,0]:.2f}, {avg_predicted_locs[i,1]:.2f}), "
            f"Error={err:.3f} m")

    # 平均誤差（全点平均）
    mean_error = np.mean(errors)
    print(f"\nAverage localization error: {mean_error:.3f} m")

    # --- 最後のEpochの擬似ラベル付きRSLデータをCSVで保存 (ここから追加) ---
    print("\nSaving last epoch pseudo-labeled data to CSV...")
    
    rss_cols = data_scalers['rss_cols']
    loc_cols = data_scalers['loc_cols']
    loc_scaler = data_scalers['loc_scaler']
    rss_scaler = data_scalers['rss_scaler']
    
    all_data = []

    for rss_tensor, pseudo_loc_tensor in last_epoch_pseudo_data:
        # 1. RSS特徴量を逆正規化
        rss_unscaled = rss_scaler.inverse_transform(rss_tensor.numpy().reshape(1, -1)).flatten()
        # 2. 擬似ラベル位置を逆正規化
        pseudo_loc_unscaled = loc_scaler.inverse_transform(pseudo_loc_tensor.numpy().reshape(1, -1)).flatten()
        
        # 3. データを結合してリストに追加
        row_data = np.concatenate([rss_unscaled, pseudo_loc_unscaled])
        all_data.append(row_data)

    if all_data:
        # DataFrameを作成
        df_cols = rss_cols + [f'pseudo_{c}' for c in loc_cols]
        df_pseudo_labeled = pd.DataFrame(all_data, columns=df_cols)
        
        # CSVファイル名を定義
        csv_filename = f'./data/pseudo_data/pseudo_labeled_data_last_epoch_train_{train_scene}_{train_date}_test_{test_scene}_{test_date}_rsl.csv'
        df_pseudo_labeled.to_csv(csv_filename, index=False)
        
        print(f"Successfully saved {len(df_pseudo_labeled)} pseudo-labeled samples to {csv_filename}")
    else:
        print("No pseudo-labels were generated in the last epoch to save.")