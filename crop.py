import pandas as pd

filename = 'data/val_day_014_td_framed_boxed_cluster.csv'

df = pd.read_csv(filename)

df = df.loc[(df['y'] >= 150) & (df['y'] <= 550)]
df = df.loc[(df['x'] >= 300) & (df['x'] <= 1200)]
df = df[['x','y','t','p','frame','track_id']]

df.to_csv(filename[:-4]+"_crop.csv",index=False)