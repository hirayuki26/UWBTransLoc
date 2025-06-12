import h5py
import pandas as pd
import numpy as np
import os

PLACE_NAME = 'OfficeP2' # Mall, OfficeP1, OfficeP2, or OfficeP1+P2
FILE_TYPE_WHICH = 'training' # training or testing

def convert_h5_to_csv(start_num, end_num, file_path_mtloc):
    """
    指定された範囲のh5ファイルをCSVに変換します。

    Args:
        start_num (int): 変換を開始するファイル番号
        end_num (int): 変換を終了するファイル番号
        file_path_mtloc (str): h5ファイルが格納されているディレクトリパス
    """
    # CSV保存ディレクトリの作成
    output_csv_dir = os.path.join(file_path_mtloc, 'csv')
    os.makedirs(output_csv_dir, exist_ok=True)

    for i in range(start_num, end_num + 1):
        file_name = f'{PLACE_NAME}_{i}_{FILE_TYPE_WHICH}'
        h5_file_path = os.path.join(file_path_mtloc, file_name + '.h5')
        csv_file_path = os.path.join(output_csv_dir, file_name + '.csv')

        print(f"--- Converting {file_name}.h5 to CSV ---")

        if not os.path.exists(h5_file_path):
            print(f"Warning: {h5_file_path} not found. Skipping.")
            continue

        try:
            # HDF5ファイルを開く
            with h5py.File(h5_file_path, 'r') as f:
                # データ読み込み
                rssis = f['rssis'][:]        # RSSI行列
                cdns = f['cdns'][:]          # 座標行列
                bssids = f['bssids'][:]      # BSSIDリスト
                # records_nums = f['RecordsNums'][:]  # 必要であればRecordsNumsも読み込み

                print('RSSIs shape:', rssis.shape)
                print('Coordinates shape:', cdns.shape)
                print('BSSIDs shape:', bssids.shape)
                # print('RecordsNums:', records_nums)

                # bssidsはバイト列かもしれないので、文字列に変換
                bssids = [bssid.decode('utf-8') if isinstance(bssid, bytes) else bssid for bssid in bssids]

            # pandas DataFrameにする
            df_rssis = pd.DataFrame(rssis, columns=bssids)
            df_cdns = pd.DataFrame(cdns, columns=['x', 'y'])  # 座標が2次元の場合

            # 必要なら座標もDataFrameにまとめる
            df = pd.concat([df_cdns, df_rssis], axis=1)

            print(df.head())

            # CSVに保存
            df.to_csv(csv_file_path, index=False)
            print(f"Successfully converted {file_name}.h5 to {file_name}.csv\n")

        except Exception as e:
            print(f"Error processing {file_name}.h5: {e}\n")

# 設定
file_path_mtloc = f'./data/{PLACE_NAME}/'
start_number = 1  # 変換を開始するファイル番号
end_number = 24    # 変換を終了するファイル番号 (例: OfficeP1_1_training.h5 から OfficeP1_5_training.h5 まで)

# 関数を実行
convert_h5_to_csv(start_number, end_number, file_path_mtloc)