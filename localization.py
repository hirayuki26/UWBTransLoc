import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NO_SIGNAL_VALUE = -100.0
QUANTITATIVE_COLUMNS = ['x', 'y']

# --- 1. データセットの準備とデータローダーの定義 ---
class WiFiDataset(Dataset):
    def __init__(self, data_df, rss_cols, loc_cols, transform=None, target_transform=None):
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
        
        return torch.tensor(rss, dtype=torch.float32), torch.tensor(loc, dtype=torch.float32)
    
def align_dataframe_columns(df, target_cols, fill_value=NO_SIGNAL_VALUE):
    current_cols = set(df.columns)
    missing_cols = [col for col in target_cols if col not in current_cols]
    
    for col in missing_cols:
        df[col] = fill_value
    
    return df[target_cols]

def create_dataloaders_for_localization(source_train_path, test_path):
    loc_cols = QUANTITATIVE_COLUMNS

    # 全てのデータファイルを読み込み、全てのAP列名を収集する
    source_train_raw = pd.read_csv(source_train_path)
    test_raw = pd.read_csv(test_path)

    all_ap_cols_set = set()
    for df in [source_train_raw, test_raw]:
        current_ap_cols = [col for col in df.columns if col not in loc_cols]
        all_ap_cols_set.update(current_ap_cols)
    
    all_rss_cols = sorted(list(all_ap_cols_set)) # APの列名をソートして固定順にする

    # 各データセットを整形する (AP列の統一と欠損値の埋め合わせ)
    source_train_aligned = align_dataframe_columns(source_train_raw.copy(), all_rss_cols + loc_cols)
    test_aligned = align_dataframe_columns(test_raw.copy(), all_rss_cols + loc_cols)

    # スケーリングのためのデータ結合
    all_rss_data = pd.concat([source_train_aligned[all_rss_cols], test_aligned[all_rss_cols]])
    all_loc_data = pd.concat([source_train_aligned[loc_cols], test_aligned[loc_cols]])

    # スケーラー初期化
    rss_scaler = MinMaxScaler(feature_range=(-1, 1))
    loc_scaler = MinMaxScaler(feature_range=(0, 1))

    # スケーラーをデータ全体でフィット
    rss_scaler.fit(all_rss_data)
    loc_scaler.fit(all_loc_data)

    # 変換定義
    rss_transform = lambda x: rss_scaler.transform(x.reshape(1, -1)).flatten()
    loc_transform = lambda y: loc_scaler.transform(y.reshape(1, -1)).flatten()

    # データセット作成 (テストデータもWiFiDatasetでラップ)
    source_train_dataset = WiFiDataset(source_train_aligned, all_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)
    test_dataset = WiFiDataset(test_aligned, all_rss_cols, loc_cols, transform=rss_transform, target_transform=loc_transform)

    # データローダー作成
    batch_size = 64
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_scalers = {
        'rss_scaler': rss_scaler,
        'loc_scaler': loc_scaler,
        'rss_cols': all_rss_cols,
        'loc_cols': loc_cols
    }

    return source_train_loader, test_loader, data_scalers

# --- 2. 測位用ネットワークアーキテクチャ定義 ---

# ConvBlk (CNN Block) - TransLocと同様
class ConvBlk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Feature Extractor - TransLocと同様だが、単体で学習
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, z_dim=16):
        super(FeatureExtractor, self).__init__()
        self.fc_initial = nn.Linear(input_dim, 1024)
        self.conv_blk1 = ConvBlk(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_blk2 = ConvBlk(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc_final = nn.Linear(32 * 8 * 8, z_dim)

    def forward(self, x):
        x = self.fc_initial(x)
        x = x.view(-1, 1, 32, 32)
        x = self.conv_blk1(x)
        x = self.conv_blk2(x)
        x = x.view(x.size(0), -1)
        z = self.fc_final(x)
        return z

# Location Predictor - TransLocと同様のTri-net構造
class LocationPredictor(nn.Module):
    def __init__(self, z_dim=16, output_loc_dim=2):
        super(LocationPredictor, self).__init__()
        # R_c (shared module)
        self.rc_fc1 = nn.Linear(z_dim, 1024)
        self.rc_conv_blk = ConvBlk(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.rc_fc2 = nn.Linear(16 * 16 * 16, 2048)

        # R_1, R_2, R_3 (parallel sub-modules)
        self.r1_conv_blk = ConvBlk(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.r1_fc1 = nn.Linear(16 * 4 * 4, 2048)
        self.r1_fc2 = nn.Linear(2048, output_loc_dim)

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
        
        rc_features_reshaped = rc_features.view(rc_features.size(0), 32, 8, 8)

        # R_1 branch
        r1_out = self.r1_conv_blk(rc_features_reshaped)
        r1_out = r1_out.view(r1_out.size(0), -1)
        r1_out = self.r1_fc1(r1_out)
        r1_pred = self.r1_fc2(r1_out)

        # R_2 branch
        r2_out = self.r2_conv_blk(rc_features_reshaped)
        r2_out = r2_out.view(r2_out.size(0), -1)
        r2_out = self.r2_fc1(r2_out)
        r2_pred = self.r2_fc2(r2_out)

        # R_3 branch
        r3_out = self.r3_conv_blk(rc_features_reshaped)
        r3_out = r3_out.view(r3_out.size(0), -1)
        r3_out = self.r3_fc1(r3_out)
        r3_pred = self.r3_fc2(r3_out)
        
        # Average of predictions for final output (Eq 9 in TransLoc paper)
        avg_pred = (r1_pred + r2_pred + r3_pred) / 3

        return avg_pred, r1_pred, r2_pred, r3_pred

# 測位モデル (FeatureExtractorとLocationPredictorの組み合わせ)
class LocalizationModel(nn.Module):
    def __init__(self, input_dim, z_dim, output_loc_dim):
        super(LocalizationModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, z_dim).to(device)
        self.location_predictor = LocationPredictor(z_dim, output_loc_dim).to(device)

    def forward(self, x):
        z = self.feature_extractor(x)
        avg_pred, r1_pred, r2_pred, r3_pred = self.location_predictor(z)
        return avg_pred, r1_pred, r2_pred, r3_pred


# --- 3. 損失関数定義 ---
location_criterion = nn.MSELoss() # 回帰タスクのためMSEを使用

# --- 4. 学習ループ ---
def train_localization_model(model, source_train_loader, test_loader, data_scalers, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test_error_history = []
    test_error_epochs_recorded = []

    print("--- Training Localization Model ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for source_rss, source_loc in source_train_loader:
            source_rss, source_loc = source_rss.to(device), source_loc.to(device)

            optimizer.zero_grad()
            
            # 測位部分
            predicted_loc_avg, predicted_loc_r1, predicted_loc_r2, predicted_loc_r3 = model(source_rss)
            
            # 損失計算 (L_R^s)
            loss = location_criterion(predicted_loc_avg, source_loc) + \
                   (location_criterion(predicted_loc_r1, source_loc) + \
                    location_criterion(predicted_loc_r2, source_loc) + \
                    location_criterion(predicted_loc_r3, source_loc)) / 3
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss / len(source_train_loader):.4f}")

        # テストセットでの評価 (測位精度)
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1: # 10エポックごと、または最終エポックで評価
            model.eval()
            total_test_distance = 0
            with torch.no_grad():
                for test_rss, test_loc in test_loader:
                    test_rss, test_loc = test_rss.to(device), test_loc.to(device)
                    
                    # 測位部分
                    predicted_loc_avg, _, _, _ = model(test_rss)
                    
                    # 予測位置を元のスケールに戻す
                    predicted_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(predicted_loc_avg.cpu().numpy())
                    true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(test_loc.cpu().numpy())

                    # ユークリッド距離で誤差を計算 (m単位)
                    distances = np.sqrt(np.sum((predicted_loc_unscaled - true_loc_unscaled)**2, axis=1))
                    total_test_distance += np.sum(distances)
            
            avg_test_distance = total_test_distance / len(test_loader.dataset)
            test_error_history.append(avg_test_distance)
            test_error_epochs_recorded.append(epoch + 1)
            print(f"--- Epoch {epoch+1} Test Localization Error: {avg_test_distance:.4f} m --- ")
            model.train() # Set back to train mode

    print("Training finished.")
    return test_error_history, test_error_epochs_recorded


# --- 実行部分 ---
if __name__ == "__main__":
    # データローダーの作成
    # 学習にはSource Domainのデータのみを使用
    # テストには、ラベルのないターゲットドメインのRSSデータ（ただし、評価のために真のラベルは必要）
    # place_name = 'OfficeP2'

    train_scene = 'wall'
    test_scene = 'wall'
    source_train_path = f'./data/uwb/processed_uwb_full_features_data_{train_scene}_train_split.csv'
    test_path = f'./data/uwb/processed_uwb_full_features_data_{test_scene}_test_split.csv'

    source_train_loader, test_loader, data_scalers = create_dataloaders_for_localization(
        source_train_path, test_path
    )

    # 入力/出力次元の取得
    sample_rss, _ = next(iter(source_train_loader))
    input_dim = sample_rss.shape[1]
    output_loc_dim = 2 # X, Y座標

    print(f"Input Dimension (RSS features): {input_dim}")
    print(f"Output Dimension (Location features): {output_loc_dim}")
    print(f"Number of samples in source_train: {len(source_train_loader.dataset)}")
    print(f"Number of samples in test: {len(test_loader.dataset)}")

    # 測位モデルの初期化 (Generator, Discriminatorなし)
    z_dim = 16
    model = LocalizationModel(input_dim, z_dim, output_loc_dim).to(device)
    print("Localization Model initialized (without Domain Adaptation components).")
    print(model)

    # # モデルの保存 (オプション)
    torch.save(model.state_dict(), f"./output/transloc_model_train_{train_scene}test_{test_scene}_localize.pth")
    print("Model saved to transloc_model.pth")

    # # テストデータで最終評価
    model.eval()
    total_test_distance = 0
    all_true_locs = []
    all_predicted_locs = []
    with torch.no_grad():
        for test_rss, test_loc in test_loader:
            test_rss, test_loc = test_rss.to(device), test_loc.to(device)
            z_test = model.feature_extractor(test_rss)
            predicted_loc_avg, _, _, _ = model.location_predictor(z_test)
            
            predicted_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(predicted_loc_avg.cpu().numpy())
            true_loc_unscaled = data_scalers['loc_scaler'].inverse_transform(test_loc.cpu().numpy())

            distances = np.sqrt(np.sum((predicted_loc_unscaled - true_loc_unscaled)**2, axis=1))
            total_test_distance += np.sum(distances)

            all_true_locs.extend(true_loc_unscaled)
            all_predicted_locs.extend(predicted_loc_unscaled)
    
    avg_test_distance = total_test_distance / len(test_loader.dataset)
    # print(f"\nFinal Test Localization Error: {avg_test_distance:.4f} m ")
    print(f"\nFinal Test Localization Error: {avg_test_distance} m ")
    with open(f"./output/localization_error_test_localize.txt", "a", encoding="utf-8") as f:
        # print(f"\ntrain_{train_scene}_test_{test_scene}\nFinal Test Localization Error: {avg_test_distance:.4f} m ", file=f)
        print(f"\ntrain_{train_scene}_test_{test_scene}\nFinal Test Localization Error: {avg_test_distance} m ", file=f)

    # Plotting Test Localization Error
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_err_epochs, test_err_hist, marker='o', linestyle='-', color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Localization Error (m)')
    # plt.title(f'Localization Error Over Epochs ( Train:{train_scene}, Test:{test_scene})')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'localization_error_plot_train{train_scene}_test{test_scene}_localize.png'))
    # plt.close()
    # print(f"Plot generated: {os.path.join(output_dir, f'localization_error_plot_train{train_scene}_test{test_scene}.png')}")

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
    plt.savefig(f'./output/final_localization_avg_plot_train_{train_scene}_test_{test_scene}_localize.png')
    plt.close()
    print(f"Final localization average plot generated: final_localization_avg_plot_train_{train_scene}_test_{test_scene}_localize.png")
