import pandas as pd
filename = 'data/val_day_014_td_framed_boxed_cluster_crop_predict2-500.csv'
df = pd.read_csv(filename)

df['correct'] = df['track_id'] == df['pred_id']
df['none'] = df['track_id'] == -1
df['TN'] = df['correct'] & df['none']
correct = df['correct'].sum()
total = len(df)
negative = df['none'].sum()
positive = total - negative
true_neg = df['TN'].sum()
false_pos = negative - true_neg
true_pos = correct - true_neg
false_neg = positive - true_pos
print(f"False negative = {false_neg/positive}  ({false_neg}/{positive})")
print(f"False positive = {false_pos/negative}  ({false_pos}/{negative})")
print(f"Correct = {correct/total}  ({correct}/{total})")