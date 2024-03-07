#import libraries
import numpy as np
import scipy as sp
import ants
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pca import pca
from matplotlib import cm
from matplotlib import style 
from scipy.stats import spearmanr


# function used to load in mri data
def get_maps(maps,samples):
#directories and files for analysis
    indir = '/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/DATA_FILES/'
    maskdir = '/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/masks_HIP/'
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
    new_mask_AD = ants.get_mask(mask_read_AD, low_thresh = low_thresholds,high_thresh = high_thresholds, cleanup=0)

#apply masks to images and resolve any incorrect values
    AD_data = ants.image_list_to_matrix(AD_file,new_mask_AD)

#format data for analysis
    data = np.hstack(AD_data)   
    return(data)

# Function to calculate Spearman's correlation and plot full heatmap
def spearmans_correlation(dataframe,color): 
    dataframe = dataframe.iloc[:,1:]
    rank_s = dataframe.corr(method='spearman')
    plt.figure(figsize=(12,8))
    sns.heatmap(rank_s,vmin=-1,vmax=1,cmap=color,annot=True)
    plt.title("Spearman Correlation of Hippocampus")
    plt.show(block=False)
    plt.savefig('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/spearman_correlation_hip_readspecimens_withoutTOTAL.png')
    return(rank_s)

#spearmans correlation plot of metrics only for pca input
def pca_spearmans(dataframe): 

    scaler = StandardScaler()
    df_new = dataframe
    #df_new = df_new.drop('Samples',axis=1)
    standard_df = pd.DataFrame(scaler.fit_transform(df_new),columns=df_new.columns)
    corr_matrix = standard_df.corr(method='spearman')
    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix,vmin=-1,vmax=1,cmap='PiYG',annot=True)
    plt.title("Spearman Correlation of Hippocampus")
    plt.show(block=False)
    plt.savefig('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/PCA_spearmans_corr_hip_readspecimens.png')
    return(corr_matrix)

# Function to create a scree plot
def scree_plot(pca):
    explained_variance_ratio_percentage = 100 * pca.explained_variance_ratio_
# Create the scree plot
    style.use('default')
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(1, pca.n_components_ + 1), explained_variance_ratio_percentage, alpha=0.6, color='blue')
# Draw a line to connect the tops of the bars
    plt.plot(range(1, pca.n_components_ + 1), explained_variance_ratio_percentage, marker='o', color='black')
# Annotate each bar with the percentage of variance explained
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variance_ratio_percentage[i]:.2f}%', ha='center', va='bottom')
    plt.xlabel('Dimensions')
    plt.ylabel('Percentage of Explained Variance')
    plt.title('Scree Plot')
    plt.xticks(range(1, pca.n_components_ + 1))
    plt.show(block=False)
    plt.savefig('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/PCA_scree_plot_hip_readspecimens.png')

# Function to create a biplot
def biplot(pca_result,dataframe):
    contribution = pca.components_.T

    # Create a biplot
    dataframe = dataframe.iloc[:,1:]
    style.use('dark_background')
    plt.figure(figsize=(16, 10))
    scaler = StandardScaler()
    standard_df = pd.DataFrame(scaler.fit_transform(dataframe),columns=dataframe.columns)
    corr_matrix = standard_df.corr(method='spearman')
    
    color_map_corr = corr_matrix.iloc[3:,0]
    color_map_corr_normalized = (color_map_corr +1)/2
    cmap = plt.cm.PiYG
    
    
    # Plot the loadings
    for i, var in enumerate(metrics):
        color = cmap(color_map_corr_normalized[i])
        plt.arrow(0, 0, contribution[i, 0], contribution[i, 1], head_width=0.02, head_length=0.02, fc=color, ec=color)
        plt.text(contribution[i, 0] *1.12, contribution[i, 1] *1.12, var, color='white', ha='center', va='center', fontsize=16)

   
    plt.gca().set_aspect('equal')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim((-0.5,0.5))
    plt.ylim((-0.6,0.6))
    plt.grid(True)
    plt.title('PCA Biplot')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  # Dummy array for the colorbar
    sm.set_clim(-1, 1)
    plt.colorbar(sm, label='Braak Score Correlation')
    plt.show(block=False)
    plt.savefig('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/PCA_biplot_hip_readspecimen.png')

# Function to create scatter plot
def plot_data(df,corr_df,CI,x_col,y_col):
    rho, pval = spearmanr(df[x_col],df[y_col])
    plt.style.use('default')
    plt.figure(figsize=(10,8))
    plt.scatter(data = df,x =x_col,y =y_col,marker="H",s=200, c=['black','red','black','red','red','red','blue','blue','blue','red','black'],alpha=0.7)
    plt.xlim([0,6.5])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Spearman Corr'+' rho ='+str(corr_df.loc[x_col,y_col])+' and CI ='+str(CI))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize=15)
    plt.savefig('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/'+y_col+'_scatter.png')

#Make dataframe of all MAPS (columns) and samples (rows). 
MAPS = ['TR','FA','MSD','MSK','PA','NG','RTOP','RTAP','RTPP','mse_gmT2','registered_sir_T1','registered_sir_BPF','mse_MWF']
SAMPLES = ['08-31','08-71','11-25','11-76','13-03','14-13','14-50','15-40','16-35','17-32','18-66']
BRAAK = np.asarray([2,6,3,6,6,6,4,5,5,6,3])
PLAQUEH = np.asarray([1.5,3,0,1.5,3,3,1.5,2,2.5,1.5,3])
TANGLEH = np.asarray([0.5,3,1.5,3,3,3,2,2,3,3,1])

NaN = 0
NAN = np.zeros(shape=11,)
dictionary = [SAMPLES,BRAAK,PLAQUEH,TANGLEH,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN]

# initialize dataframe
df = pd.DataFrame()
df = pd.DataFrame(dictionary).T
df.columns = ['Samples','Braak Score','Hip Plaque','Hip Tangle','TR','FA','MSD','MSK','PA','NG','RTOP','RTAP','RTPP','gmT2','T1','BPF','MWF']

#for loop for populating dataframe
for i, maps in enumerate(MAPS):
    for j, samples in enumerate(SAMPLES):
        temp_array = get_maps(maps,samples)
        mean = np.nanmean(temp_array,dtype=np.float64)
        if maps == 'registered_sir_BPF':
            df.loc[j,'BPF'] = mean
        elif maps == 'mse_gmT2':
            df.loc[j,'gmT2'] = mean
        elif maps == 'mse_MWF':
            df.loc[j,'MWF'] = mean
        elif maps == 'registered_sir_T1':
            df.loc[j,'T1'] = mean
        else:
            df.loc[j,maps] = mean

df.dtypes

#correct the dataframe format to float64 type for future calculations
columns_to_convert = ['Braak Score','Hip Plaque','Hip Tangle','TR','FA','MSD','MSK','PA','NG','RTOP','RTAP','RTPP','gmT2','T1','BPF','MWF']
for column in columns_to_convert:
    df[column] = np.asarray(df[column].values.astype(np.float64))

#Dataframe to be used for PCA 
df_new= df.iloc[:,4:]
metrics = ['TR','FA','MSD','MSK','PA','NG','RTOP','RTAP','RTPP','gmT2','T1','BPF','MWF']

#start PCA
corr_matrix= pca_spearmans(df_new)

pca = PCA()
pca_result = pca.fit_transform(corr_matrix)

#save pca loadings
loadings_2d = pca.components_[:2,:].T
loadings_df = pd.DataFrame(data=loadings_2d,columns=['PC1','PC2'],index=metrics) 
loadings_df.to_csv('/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/pca_analysis/HIP_loadings.csc') 

#call pca plotting functions
scree_plot(pca)
biplot(pca_result,df)

#Dataframe and steps for spearmans correlation and bootstrapping
plt.style.use('default')
spear_cor = spearmans_correlation(df,'PiYG')

df_braak = df['Braak Score']

boot_corrs = np.zeros(1000)

n = len(df_braak)

# Bootstrapping for confidence interval
for i in range(1):
    df_path = df.iloc[:,i]
    for j in range(13):
        map = df.iloc[:,3+j]
        for k in range(1000):
            boot_index = np.random.choice(np.arange(n),size=n,replace=True)
            corr, _ = spearmanr(df_path[boot_index],map[boot_index])
            boot_corrs[k] = corr
            boot_findnan = boot_corrs[~np.isnan(boot_corrs)]
        confidence_interval = np.percentile(boot_findnan, [5, 95])

        # Create scatter plot
        plot_data(df,spear_cor,confidence_interval,'Braak Score',metrics[j])

input("press enter to exit")