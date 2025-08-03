import pandas as pd
from sklearn.model_selection import train_test_split

scene = 'wall_null_A'
fname = f'processed_uwb_rsl_data_{scene}'

print(fname)

# Load the data
df = pd.read_csv(f'./data/uwb/{fname}.csv')

# Identify coordinate columns
loc_cols = ['x', 'y']

# Initialize empty dataframes for train and test sets
train_df_list = []
test_df_list = []

# Group by coordinates and split
# 座標ごとにグループ化し、シーケンシャルに分割
for _, group in df.groupby(loc_cols):
    # グループ内の分割点を計算（70%をトレーニング用とする）
    split_point = int(len(group) * 0.7)

    # グループをシーケンシャルに分割し、インデックスをリセット
    train_group = group.iloc[:split_point].reset_index(drop=True)
    test_group = group.iloc[split_point:].reset_index(drop=True)

    # 分割されたグループをリストに追加
    train_df_list.append(train_group)
    test_df_list.append(test_group)

# すべてのパーツを連結して最終的なトレーニングデータとテストデータフレームを作成
train_df = pd.concat(train_df_list).reset_index(drop=True)
test_df = pd.concat(test_df_list).reset_index(drop=True)

print(len(df))
print(len(train_df))
print(len(test_df))

print(train_df.head())
print(test_df.head())

# Save the split data to new CSV files
train_output_path = fname + '_train_split.csv'
test_output_path = fname + '_test_split.csv'

train_df.to_csv(f'./data/uwb/{train_output_path}', index=False)
test_df.to_csv(f'./data/uwb/{test_output_path}', index=False)

print(f"Training data saved to: {train_output_path}")
print(f"Testing data saved to: {test_output_path}")