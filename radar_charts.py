import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import numpy as np
import scipy as sp
import ants
import scipy.stats as stats

def get_maps_char(maps,samples):
#directories and files for analysis
    indir = '/xdisk/hutchinsone/courtneycomrie/Data/Development/PM/TLobe/12_sample_dataset/ISMRM_data/'
    maskdir = '/xdisk/hutchinsone/courtneycomrie/Data/Development/PM/TLobe/12_sample_dataset/masks_HIP/'
    ext = '.nii'
    sample_AD = indir+maps+'_'+samples+ext

#thresholds for masks
    low_thresholds = np.array(0.5) 
    high_thresholds = np.array(1.5)

#read in imagefiles    
    AD_file = [ants.image_read(sample_AD)]

#read in and prep masks
    seg_AD = maskdir + samples+'_hippocampus'+ext #HIP PATTERN
    mask_read_AD = ants.image_read(seg_AD,3) 
    new_mask_AD = ants.get_mask(mask_read_AD, low_thresh = low_thresholds,high_thresh = high_thresholds,cleanup=0)

#apply masks to images and resolve any incorrect values
    AD_data = ants.image_list_to_matrix(AD_file,new_mask_AD)

#format data for analysis
    data = np.hstack(AD_data)   
    return(data)

def build_df_relax(SAMPLES):
    MAPS = ['FA','TR','registered_sir_BPF','mse_gmT2','mse_MWF','registered_sir_T1'] #we will add BPF,MWF, T2, and T2* later
    samples = SAMPLES
    groups = ['early','mid','late']

    NaN = 0
    NAN = np.zeros(shape=2,)
    dictionary = [groups,NAN,NAN,NAN,NAN,NAN,NAN]

    df = pd.DataFrame()
    df = pd.DataFrame(dictionary).T
    df.columns = ['groups','FA','TR','registered_sir_BPF','mse_gmT2','mse_MWF','registered_sir_T1']
    # df.columns = ['groups']
    #for loop for populating dataframe
   
    for i, maps in enumerate(MAPS):
        arr_0831 = get_maps_char(maps,samples[0])
        arr_0871 = get_maps_char(maps,samples[1])
        arr_1125 = get_maps_char(maps,samples[2])
        arr_1176 = get_maps_char(maps,samples[3])
        arr_1303 = get_maps_char(maps,samples[4])
        arr_1413 = get_maps_char(maps,samples[5])
        arr_1450 = get_maps_char(maps,samples[6])
        arr_1540 = get_maps_char(maps,samples[7])
        arr_1635 = get_maps_char(maps,samples[8])
        arr_1732 = get_maps_char(maps,samples[9])
        arr_1866 = get_maps_char(maps,samples[10])

        arr_early = np.concatenate((arr_0831,arr_1125,arr_1866))
        arr_mid = np.concatenate((arr_1450,arr_1540,arr_1635))
        arr_late = np.concatenate((arr_0871,arr_1176,arr_1303,arr_1413,arr_1732))

        mean_early = np.nanmean(arr_early)
        mean_mid = np.nanmean(arr_mid)
        mean_late = np.nanmean(arr_late)

        df.loc[0,maps] =mean_early
        df.loc[1,maps] =mean_mid
        df.loc[2,maps] =mean_late
    df.rename(columns={'registered_sir_BPF':'BPF','mse_gmT2':'gmT2','mse_MWF':'MWF','registered_sir_T1':'T1'},inplace=True)
    return(df)

def build_df_restrict(SAMPLES):
    MAPS = ['MSK','MSD','RTOP','NG','TR'] #we will add BPF,MWF, T2, and T2* later
    samples = SAMPLES
    groups = ['early','mid','late']

    NaN = 0
    NAN = np.zeros(shape=2,)
    dictionary = [groups,NAN,NAN,NAN, NAN, NAN]

    df = pd.DataFrame()
    df = pd.DataFrame(dictionary).T
    df.columns = ['groups','MSK','MSD','RTOP','NG','TR']
  
    #for loop for populating dataframe
   
    for i, maps in enumerate(MAPS):
        arr_0831 = get_maps_char(maps,samples[0])
        arr_0871 = get_maps_char(maps,samples[1])
        arr_1125 = get_maps_char(maps,samples[2])
        arr_1176 = get_maps_char(maps,samples[3])
        arr_1303 = get_maps_char(maps,samples[4])
        arr_1413 = get_maps_char(maps,samples[5])
        arr_1450 = get_maps_char(maps,samples[6])
        arr_1540 = get_maps_char(maps,samples[7])
        arr_1635 = get_maps_char(maps,samples[8])
        arr_1732 = get_maps_char(maps,samples[9])
        arr_1866 = get_maps_char(maps,samples[10])

        arr_early = np.concatenate((arr_0831,arr_1125,arr_1866))
        arr_mid = np.concatenate((arr_1450,arr_1540,arr_1635))
        arr_late = np.concatenate((arr_0871,arr_1176,arr_1303,arr_1413,arr_1732))

        mean_early = np.mean(arr_early)
        mean_mid = np.mean(arr_mid)
        mean_late = np.mean(arr_late)

        df.loc[0,maps] =mean_early
        df.loc[1,maps] =mean_mid
        df.loc[2,maps] =mean_late

    return(df)

SAMPLES = ['08-31','08-71','11-25','11-76','13-03','14-13','14-50','15-40','16-35','17-32','18-66']

df_map = build_df_relax(SAMPLES)
#df_map = build_df_restrict(SAMPLES)

categories = ['FA','TR','BPF','gmT2','MWF', 'T1']
#categories = ['MSK','MSD','RTOP','NG','TR']
categories = [*categories, categories[0]]

fig = go.Figure()

early = []
mid = []
late = []

for i in range(1,6):
    early_val = df_map.iloc[0,i]
    mid_val = df_map.iloc[1,i]
    late_val = df_map.iloc[2,i]

    mid_val = mid_val / early_val
    late_val = late_val / early_val
    early_val = early_val / early_val

    early.append(early_val)
    mid.append(mid_val)
    late.append(late_val)


early = [*early,early[0]]
mid = [*mid,mid[0]]
late = [*late,late[0]]

fig = go.Figure(data=[go.Scatterpolar(r=early,marker=dict(color='rgb(0,0,0)'),opacity =0.5,theta=categories,fill='toself',name='BRAAK II-III'),
                      go.Scatterpolar(r=mid,marker=dict(color='rgb(0,0,255)'),opacity =0.5,theta=categories,fill='toself',name='BRAAK IV-V'),
                      go.Scatterpolar(r=late,marker=dict(color='rgb(255,0,0)'),opacity =0.5,theta=categories,fill='toself',name='BRAAK VI')],
        layout=go.Layout(
        title=go.layout.Title(text='11 Sample Dataset'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True))
fig.show()
#fig.write_image('/xdisk/hutchinsone/courtneycomrie/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/radar_plots/relax_comparison_11samples_new.png')