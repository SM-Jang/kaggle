import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

def make_image(df):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in df['band_1']])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in df['band_2']])
    
    image = np.concatenate([X_band_1[:,:,:,np.newaxis], 
                    X_band_2[:,:,:,np.newaxis], 
                   ((X_band_1+X_band_2)/2)[:,:,:,np.newaxis]], axis=-1)
    return image

def getData(df, dtype=None):
    y = None
    # train/valid split
    if dtype=='train':
        y = df['is_iceberg'].to_numpy()
        
    X = make_image(df)
    X = X.transpose(0,3,1,2)
    return X,y


def plotmy3d(c, name):
    data=[go.Surface(z=c)]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(l=65,r=50,b=65,t=90)
    )
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig)
    
    