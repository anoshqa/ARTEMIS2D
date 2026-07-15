import os
import numpy as np
import pandas as pd


def get_base_name(filename):
    filename = str(filename)
    if filename.endswith('_rgb.png'):
        return filename[:-9]
    if filename.endswith('.tiff'):
        return filename[:-6]
    return os.path.splitext(filename)[0]


prop = pd.read_csv('hc_features.csv')
propdino = pd.read_csv('dino_features_mip_masked.csv')

prop = prop.dropna(axis=0, how='any').reset_index(drop=True)
prop['mask_key'] = prop['mask_file'].astype(str).apply(get_base_name)
propdino['mask_key'] = propdino['mask_file'].astype(str).apply(get_base_name)

prop['row_index'] = np.arange(len(prop))
propdino['row_index'] = np.arange(len(propdino))

matched_dino = []
for _, row in prop.iterrows():
    key = row['mask_key']
    candidates = propdino[propdino['mask_key'] == key]
    if not candidates.empty:
        matched_row = candidates.iloc[0]
    else:
        matched_row = propdino.iloc[row['row_index']] if row['row_index'] < len(propdino) else pd.Series(dtype='object')
    matched_dino.append(matched_row)

matched_dino = pd.DataFrame(matched_dino).reset_index(drop=True)
matched_dino = matched_dino.drop(columns=['mask_key', 'row_index'], errors='ignore')

propnumeric = prop.drop(columns=['Type', 'image_file', 'mask_file', 'mask_key', 'row_index'])
masklist = prop['mask_file'].tolist()
supervised_labels = np.array(prop['Type'])

print('NaN count after cleaning:', propnumeric.isna().sum().sum())
prop.to_csv('hc_features_cleaned.csv', index=False)

matched_dino.to_csv('dino_features_mip_cleaned.csv', index=False)

print(f'Cleaned {len(prop)} handcrafted rows and {len(matched_dino)} DINO rows.')