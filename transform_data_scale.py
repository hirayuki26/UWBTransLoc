import pandas as pd

scene = 'non_obst'
fname = f'processed_uwb_full_features_data_{scene}'

# CSVファイルをロードします
df = pd.read_csv(f'./data/pre-uwb/{fname}.csv')

# x座標とy座標を0.5倍に変換します
df['x'] = df['x'] * 0.5
df['y'] = df['y'] * 0.5

# 変換されたデータを新しいCSVファイルに保存します
output_file_name = f'./data/uwb/{fname}.csv'
df.to_csv(output_file_name, index=False)

print(f"変換されたデータは {output_file_name} に保存されました。")