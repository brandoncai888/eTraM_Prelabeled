import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time


def color(idx):
    if idx == -1:
        return 'black'
    cmap = matplotlib.colormaps['tab20']
    
    return cmap(idx % 20)

def display(eventsFile,bboxFile,boxOn,cropped,id):
    print("Plotting 2D frames")

    print(f"Reading {eventsFile} ...")
    t1 = time.time()
    events = pd.DataFrame()
    if eventsFile[-4:] == ".csv":
        events = pd.read_csv(eventsFile)
    if eventsFile[-8:] == ".parquet":
        events = pd.read_parquet(eventsFile)
    print(time.time()-t1)
    print(len(events))

    print(f"Reading {bboxFile} ...")
    t1 = time.time()
    bboxs = pd.read_csv(bboxFile)
    print(time.time()-t1)


    fig,ax = plt.subplots(figsize=(9,4))
    if cropped:
        xmax = 1200
        ymax = 550
        xmin = 300
        ymin = 150

    else:
        xmax = 1279
        ymax = 719
        xmin = 0
        ymin = 0
        fig.set_size_inches(9.6, 5.4) 

    
    for i,events_t in events.groupby('frame'):
        #i = int(i)
        bbox = bboxs[bboxs['frame']==i]
        ax.clear()
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymax,ymin)
        try:
            for j,events_track in events_t.groupby(id):
                j = int(j)
                events_track.plot.scatter(x='x',y='y',ax=ax,color=color(j), s=0.03)
        except:
            events_t.plot.scatter(x='x',y='y',ax=ax,color=color(-1), s=0.03)
        
        if boxOn:
            for _, row in bbox.iterrows():
                rect = patches.Rectangle(
                    (row['x'], row['y']), 
                    row['w'], 
                    row['h'], 
                    linewidth=1, 
                    edgecolor=color(row["track_id"].astype(int)), 
                    facecolor='none'
                )
                ax.add_patch(rect)
        

        plt.draw()
        plt.pause(0.001)
        print(f"Frame {i}")

def plot_3d(eventsFile,bboxFile):
    print("Plotting 3D (x,y,t)")

    # 1. Load the data
    print(f"Reading {eventsFile} ...")
    t1 = time.time()
    events = pd.read_parquet(eventsFile)
    print(time.time()-t1)

    print(f"Reading {bboxFile} ...")
    t1 = time.time()
    bboxs = pd.read_csv(bboxFile)
    print(time.time()-t1)

    t_steps = bboxs['t'].unique()
    events['range'] = pd.cut(events['t'],bins=t_steps,labels=False,include_lowest=True)

    #crop events as desired
    t_min = 150
    t_max = 350
    df = events.loc[(events['range']>=t_min) & (events['range']<t_max)]
    df = df.loc[(df['x'] < 1250) & (df['x'] > 250) & (df['y'] < 550) & (df['y'] > 150)]
    print(len(df))

    # 2. Setup the 3D plot
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10,azim=-25,roll=0)
    ax.set_box_aspect((4, 2, 1))
    ax.set_xlim(t_steps[t_min],t_steps[t_max])
    ax.set_ylim(250,1250)
    ax.set_zlim(550,150)


    t1  = time.time()
    # 3. Group by 'id' and plot each group separately
    # This is efficient because it minimizes the number of plot objects created
    for identity, group in df.groupby('track_id'):
        ax.scatter(
            group['t'], 
            group['x'], 
            group['y'], 
            color=color(identity), 
            #label=f'ID {identity}',
            s=.01,  
            alpha=0.15
        )

    # 4. Labeling
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    # Optional: Add legend (be careful if you have hundreds of IDs)
    # ax.legend()

    plt.savefig(f"images/3d graphs/val_day_014_[{t_min},{t_max}].png", dpi=300)
    print(f"{time.time()-t1} seconds to plot {len(df)} points")
    plt.show()



if __name__ == "__main__":
    folder = './data'
    eventsFile = folder+'/'+'val_day_014_td_framed_boxed_cluster_crop_predict2-500.csv'
    bboxFile = folder+'/'+'val_day_014_bbox_framed.csv'
    #plot_3d(eventsFile,bboxFile)
    display(eventsFile,bboxFile,True,False,'pred_id')
'''
t1 = time.time()
df = pd.read_csv(eventFilename)
start = min(df['t'])
end = max(df['t'])
xmax = max(df['x'])
ymax = max(df['y'])
xmin = min(df['x'])
ymin = min(df['y'])
df['y'] = ymax+ymin-df['y']
print(f"t: {start} - {end} \nx: {xmin} - {xmax} \ny: {ymin} - {ymax} \n")
print(f"Loaded {eventFilename} in {time.time()-t1} seconds")
print(df.head())

dfBox = pd.read_csv(boxFilename)
dfBox['y'] = ymax+ymin-dfBox['y']
dfBox['h'] = -dfBox['h']

# # time offset in 007 for some reason
# df['t'] = df['t'] - min(df['t']) 
# dfBox['t'] = dfBox['t'] - min(dfBox['t'])

print("generating frames...")
t1 = time.time()
dfList = []
dfBoxList = []
for i in range(start,end,delta):
    dfList.append((df[(df['t'] >= i) & (df['t'] < i+delta)]))
    dfBoxList.append((dfBox[(dfBox['t']>=i)&(dfBox['t']<i+delta)]))
print(f"{len(dfList)} frames generated in {time.time()-t1} seconds")

fig, ax = plt.subplots()
total = 0
count = 0
for i in range(len(dfList)):
    t1 = time.time()
    ax.clear()
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    df = dfList[i]
    frame = df[df['p'] == 1]
    frame.plot.scatter(x='x',y='y',ax=ax,color='red', s=0.033)

    frame = df[df['p'] == 0]
    frame.plot.scatter(x='x',y='y',ax=ax,color='blue',title=f"frame {i}",s=0.033)

    dfBox = dfBoxList[i]
    for index, row in dfBox.iterrows():
        # Create a Rectangle patch
        # patches.Rectangle((x, y), width, height)
        rect = patches.Rectangle(
            (row['x'], row['y']), 
            row['w'], 
            row['h'], 
            linewidth=2, 
            edgecolor='black', 
            facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)



    plt.draw()
    plt.pause(0.0001)
    count += 1
    total += (time.time()-t1)
    if count % 50 == 0:
        print(f"{round(count/total,2)} Hz")
'''