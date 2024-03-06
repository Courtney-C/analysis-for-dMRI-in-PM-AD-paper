%% Setup for directories and specific map
close all; clear all;
main_indir='/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/ISMRM_data/';
main_outdir='/xdisk/hutchinsone/cjoy1895/Data/Development/PM/TLobe/12_sample_dataset/data_analysis/averaged_histograms/';

map = 'PA_';
%% Get data for hippocampus
% hippocampus specimen IDs
samples = {'08-31','08-71','11-25','11-76','13-03','14-13','14-50','15-40','16-35','17-32','18-66'};

% Initialize a cell array to store the masks
mask_array = cell(size(samples));

% Iterate over each sample
for idx = 1:length(samples)
    % Generate the filename for the current sample
    filename = fullfile(main_indir, [samples{idx} '_hippocampus.nii']);
    
    % Read the corresponding mask and store it in the masks cell array
    mask_array{idx} = niftiread(filename);
end

%set up and fill data arrays
%mask_array={mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8,mask_9,mask_10,mask_11};
masked_sample = {};
hist_array={};
centers_array={};
 
for i = 1:length(samples)
     sample_num=samples{i};
     current_sample = niftiread(fullfile(main_indir,append(map,sample_num)));
     masked_sample{i}=current_sample(find(mask_array{i}));

     [hist,centers]=histnorm(masked_sample{i},100,'BinLimits',[0, 1]);
     hist_array{i} = hist; centers_array{i}=centers;   
 end

% groupings for braak staging
ave_early=mean([hist_array{1};hist_array{3};hist_array{11}]);
ave_mid = mean([hist_array{7};hist_array{8};hist_array{9}]);
ave_late = mean([hist_array{2};hist_array{4};hist_array{5};hist_array{6};hist_array{10}]);


%% Plot Data
centers = centers_array{1};

figure(1)
plot(centers, ave_early,'-k','LineWidth',2);
hold on 
plot(centers, ave_mid,'-b','LineWidth',2)
plot(centers, ave_late,'-r','LineWidth',2)
legend('BRAAK II-III', 'BRAAK IV-V','BRAAK VI' )
hold off

%save figure
saveas(figure(1), fullfile(main_outdir,'PA_ave_hist_EC.png'))