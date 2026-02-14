import h5py
import pandas as pd
import numpy as np
import os

def extract_h5_to_csv(h5_input_path, csv_output_path):
    with h5py.File(h5_input_path, 'r') as f:
        if 'events' in f:
            group = f['events']
            print(list(f['events'].keys()))
            data = {
                't': group['t'][:],
                'x': group['x'][:],
                'y': group['y'][:],
                'p': group['p'][:]
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_output_path, index=False)
            print(f"Extraction complete! Saved {len(df)} events to {csv_output_path}")
        else:
            print(f"Key 'events' not found. Available keys: {list(f.keys())}")

def decode_npy_to_csv(npy_input_path, csv_output_path):
    data = np.load(npy_input_path)
    print(data.dtype.names)
    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    print(f"Success! Decoded {len(df)} rows to {csv_output_path}")


if __name__ == "__main__":
    folder = './data'
    for filename in ["val_day_014_td.h5","val_day_014_bbox.npy"]:  #os.listdir(folder):
        if filename[-3:] == '.h5':
            filename = filename[:-3]
            extract_h5_to_csv(folder+"/"+filename+".h5",folder+"/"+filename+".csv")
        elif filename[-4:] == '.npy':
            filename = filename[:-4]
            decode_npy_to_csv(folder+"/"+filename+".npy",folder+"/"+filename+".csv")
