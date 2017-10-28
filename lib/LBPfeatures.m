function LBPfeatures()
conf.calDir = '../' ; % calculating directory
conf.dataDir = '../data/training_set/train/' ; % data (image) directory 
conf.outDir = '../output/'; % output directory
conf.prefix = 'lbp_' ;
conf.lbpPath = fullfile(conf.outDir, [conf.prefix 'feature.mat']);

imageSet = dir(strcat(conf.dataDir,'*.jpg'));    
imageNum = length(imageSet);
lbp = zeros(imageNum, 59);

for i = 1:imageNum
    img = imread(fullfile(conf.dataDir,imageSet(i).name));
    if(length(size(img)) == 3)
        img = rgb2gray(img);
    end
    lbp(i,:) = extractLBPFeatures(img);
    sprintf('%s%d','image',i,'completed')
end

save(conf.lbpPath, 'lbp');
csvwrite('../data/feature_LBP.csv',lbp);
end