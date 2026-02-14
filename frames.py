import pandas as pd

def to_frames(bboxFile,framesFile,frame_size):
    print(f"Reading {bboxFile} ...")
    bboxs = pd.read_csv(bboxFile)

    # 1. Convert to a list (NumPy arrays from .unique() don't support .append)
    t_steps = sorted(bboxs['t'].unique().tolist())


    # 2. Use a while loop to handle the dynamic size of t_steps
    i = 0
    while i < len(t_steps) - 1:
        t0 = t_steps[i]
        t1 = t_steps[i+1]

        # Check if the gap is large enough to insert a new frame
        if t1 - t0 >= 2 * frame_size:
            new_t = t0 + frame_size
            
            # Insert the new value right after the current index
            t_steps.insert(i + 1, new_t)
            
            # We DON'T increment 'i' yet because we want to check the gap 
            # between the new value and the next one (t1) in the next iteration.
        
        i += 1

    # 3. Save the result
    # Convert back to Series so we can use .to_csv
    pd.Series(t_steps, name='t').to_csv(framesFile, index=False)
    print(f"Done! Saved to {framesFile}")

def frameStats(framesFile):
    frames = pd.read_csv(framesFile)['t'].to_list()
    delta = []
    for i in range(len(frames)-1):
        delta.append(frames[i+1]-frames[i])
    distinct_deltas = []
    num_deltas = []
    for d in delta:
        try:
            i = distinct_deltas.index(d)
            num_deltas[i] += 1
        except:
            distinct_deltas.append(d)
            num_deltas.append(1)
    for i in range(len(distinct_deltas)):
        print(f"{num_deltas[i]} frames of size {distinct_deltas[i]}")

def frameEvents(eventFile,framesFile,framedEventFile):
    events = pd.read_csv(eventFile)
    frames = pd.read_csv(framesFile)
    events['frame'] = pd.cut(events['t'],bins=frames['t'],labels=False,include_lowest=True)
    framedEvents = events.dropna(subset=['frame'])
    framedEvents.to_csv(framedEventFile,index=False)

if __name__=="__main__":
    eventFile = "./data/val_day_014_td.csv"
    bboxFile = "./data/val_day_014_bbox.csv"
    framesFile = "./data/val_day_014_frames.csv"
    framedEventFile = "./data/val_day_014_td_framed.csv"
    framedBboxFile = "./data/val_day_014_bbox_framed.csv"
    frame_size = 33333
    #to_frames(bboxFile,framesFile,frame_size)
    #frameStats(framesFile)
    frameEvents(eventFile,framesFile,framedEventFile)
    frameEvents(bboxFile,framesFile,framedBboxFile)