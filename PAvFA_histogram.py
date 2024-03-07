# import libraries
import ants
import numpy as np
import matplotlib.pyplot as plt

indir = '/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/ISMRM_data/'
outdir = '/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/1D_hist/'
maps ='PA_15-82' #map input
ext = '.nii'
mask1 = '15-71_hippocampus_COMRIE.nii' 
mask2 = '15-82_hippocampus_COMRIE.nii'
seg_file1 = indir + mask1
seg_file2 = indir + mask2
mask_read1 = ants.image_read(seg_file1,3)
mask_read2 = ants.image_read(seg_file2,3)
new_mask1 = ants.get_mask(mask_read1, low_thresh = 0.5, high_thresh = 1.5,cleanup=0) #get and apply thresh to masks
new_mask2 = ants.get_mask(mask_read2, low_thresh = 0.5, high_thresh = 1.5,cleanup=0)
masks = [new_mask1,new_mask2]

name_data1 = indir + maps +ext
file_data1 = [ants.image_read(name_data1)]

fig1, ax1 = plt.subplots() #plot two graphs
plt.ion()
ax1.title.set_text('PA Healthy vs AD') #graph titles
ax1.set(xlabel="Intensity",ylabel="Relative Frequency") #axis titles

facecolors = ['red','blue'] #two colors used
labels = ['Alzheimers','Healthy']
for index, mask_num in enumerate(masks): #apply ants voxel analysis
    data1 = ants.image_list_to_matrix(file_data1,mask_num)
    np_data1 = np.hstack(data1)
    ax1.hist(np_data1,facecolor=facecolors[index],bins = 150,alpha=0.5,edgecolor ='black',range=(0,0.8),label=labels[index], weights=np.zeros_like(np_data1) + 1. / np_data1.size)    

ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #apply legend
fig1.show()
plt.figure().set_figwidth(30)
plt.savefig(outdir+maps+'_PA_HE_AD.png',dpi=300, bbox_inches = "tight")
plt.ioff()