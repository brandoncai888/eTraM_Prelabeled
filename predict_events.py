import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time

def predict_events_frame(prev_frame, events_frame, spatial_threshold, time_scale, chunk_size):
    """
    Sequentially processes events in chunks. Events assigned an ID in 
    the current chunk become searchable for the next chunk.
    """
    # Ensure current frame is sorted by time for sequential logic
    events_frame = events_frame.sort_values('t').copy()
    events_frame['pred_id'] = -1

    # Initialize the searchable pool with the previous frame
    search_coords = np.column_stack((
        prev_frame['x'], 
        prev_frame['y'], 
        prev_frame['t'] / time_scale
    ))
    search_ids = prev_frame['pred_id'].values

    # Process events_frame in chunks
    # Larger chunks = faster speed; Smaller chunks = more "up-to-the-millisecond" updates
    for i in range(0, len(events_frame), chunk_size):
        chunk = events_frame.iloc[i : i + chunk_size]
        
        # Build tree from all known points (prev frame + already processed chunks)
        tree = KDTree(search_coords)
        
        # Prepare coordinates for the current chunk
        target_coords = np.column_stack((
            chunk['x'], 
            chunk['y'], 
            chunk['t'] / time_scale
        ))
        '''
        # Query the nearest neighbor
        # distance_upper_bound ensures we respect the spatial_threshold in 3D
        distances, indices = tree.query(target_coords, k=1, distance_upper_bound=spatial_threshold)

        # Map indices back to IDs
        assigned_ids = np.full(len(chunk), -1)
        valid_mask = indices < len(search_coords)
        assigned_ids[valid_mask] = search_ids[indices[valid_mask]]

        # Update the events_frame with results
        events_frame.iloc[i : i + chunk_size, events_frame.columns.get_loc('pred_id')] = assigned_ids

        # Append this chunk to the searchable pool for the NEXT chunk
        search_coords = np.vstack([search_coords, target_coords])
        search_ids = np.concatenate([search_ids, assigned_ids])
        '''
        # 1. Query the 3 nearest neighbors instead of 1
        # distance_upper_bound still respects your spatial_threshold
        distances, indices = tree.query(target_coords, k=3, distance_upper_bound=spatial_threshold)

        # 2. Map indices back to IDs 
        # Note: indices will be a 2D array of shape (len(chunk), 3)
        assigned_ids = np.full(len(chunk), -1)
        
        # We iterate through the k-neighbors (column-wise) 
        # to find the first one that isn't -1
        for neighbor_idx in range(indices.shape[1]):
            # Get the indices for this neighbor rank (0th, 1st, or 2nd)
            current_k_indices = indices[:, neighbor_idx]
            
            # Mask for valid indices (within tree bounds and distance threshold)
            valid_mask = current_k_indices < len(search_coords)
            
            # Temporary mapping to check the actual IDs
            potential_ids = np.full(len(chunk), -1)
            potential_ids[valid_mask] = search_ids[current_k_indices[valid_mask]]
            
            # Update assigned_ids ONLY where they are still -1 
            # AND the potential_id we just found is NOT -1
            update_mask = (assigned_ids == -1) & (potential_ids != -1)
            assigned_ids[update_mask] = potential_ids[update_mask]
            
            # Optimization: if all points have an assigned ID, we can stop looking at further neighbors
            if not np.any(assigned_ids == -1):
                break

        # 3. Update the events_frame with results
        events_frame.iloc[i : i + chunk_size, events_frame.columns.get_loc('pred_id')] = assigned_ids

        # 4. Append this chunk to the searchable pool for the NEXT chunk
        search_coords = np.vstack([search_coords, target_coords])
        search_ids = np.concatenate([search_ids, assigned_ids])

    return events_frame

def process_events(events,chunk_size):
    start_frames = [291, 326, 388, 467, 595, 1520]
    results = []
    
    # Starting state
    prev_frame = pd.DataFrame({
        'x': [0], 'y': [0], 't': [0], 'p': [0], 
        'frame': [-1], 'track_id': [-1], 'pred_id': [-1]
    })
    
    for frame_id, events_frame in events.groupby('frame'):
        print(f"Processing frame: {int(frame_id)}  ({len(events_frame)} events)")

        if frame_id in start_frames:
            prev_frame['pred_id'] = prev_frame['track_id']
            
        # We pass a chunk_size (e.g., 200). 
        # Increase this for more speed, decrease for more granular sequential matching.
        updated_frame = predict_events_frame(prev_frame, events_frame, 6.0, 2500, chunk_size)
        results.append(updated_frame)
        
        # The entire processed frame becomes the 'previous' for the next frame
        prev_frame = updated_frame

    return pd.concat(results, ignore_index=True)



events = pd.read_csv("data/val_day_014_td_framed_boxed_cluster_crop.csv")
#events = events.loc[(events['frame'] > 288) & (events['frame'] < 320)]
t1 = time.time()
events_track = process_events(events,500)
with open('runtime2.txt','a') as f:
    f.write(f'500 = {round(time.time()-t1,1)}s\n')
print(f"Saving data...")
events_track = events_track[['x','y','t','p','frame','track_id','pred_id']]
events_track.to_csv("data/val_day_014_td_framed_boxed_cluster_crop_predict.csv",index=False)


