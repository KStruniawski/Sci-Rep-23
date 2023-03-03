% preprocessing image and segmentation creating an image mask with image 
% scaling functionality, input images and masks paths

function makeMasks2(imagesPath, masksPath, scale)
    % no scale parameter means no scaling
    if nargin == 2
        scale = 1;
    end
        
    %open in folder
    cd(imagesPath);
    images = dir;
    cd ..;
    cd ..;
    
    %open out folder
    cd(masksPath);
    toDel = dir;
    cd ..;
    cd ..;
        
    % clear out folder
    parfor i = 1:length(toDel)
        path = toDel(i).folder + "/" + toDel(i).name;
        try
            delete(path);
        catch
            continue;
        end
    end
    
    % loop to create image mask
    for i = 1:length(images)
        % read image
        path = images(i).folder + "/" + images(i).name;
        try
            I = imread(path);
        catch
            continue;
        end
        
        % check if scaling image is necessary
        if scale ~= 1
            % resize image
            I = imresize(I,scale);
            
            % preprocess image
            I_p = preprocessingIm(I);
            
            % save to path ex. /in/scale5
            newpath = pwd + "/in/scale" + scale*10;
            mkdir(newpath);
            imwrite(I, newpath + "/" + images(i).name);
            
            % save mask to path ex. /out/scale5
            newpath = pwd + "/out/scale" + scale*10;
            mkdir(newpath);
            imwrite(I_p, newpath + "/mask_" + images(i).name,'BitDepth',1);
        else
            % preprocess image and save mask
            I_p = preprocessingIm(I);
            imwrite(I_p, masksPath + "/mask_" + images(i).name,'BitDepth',1);
        end
    end
end

function BW = preprocessingIm(I)
    map = colorChangesMap(I);
    I_bw = rgb2gray(I);
    I_bw(map) = mean(I_bw,'all');
    %I_bw = imlocalbrighten(I_bw);
    I_bw = adapthisteq(I_bw);
    I_bw = imgaussfilt(I_bw);
    I_bw = wiener2(I_bw,[3 3]);

    level = adaptthresh(I_bw, 0.92);
    BW = imbinarize(I_bw,level);
    BW = imcomplement(BW);
%     BW = imfill(BW,'holes');
%     BW = bwareaopen(BW, 100, 4);
    se = strel('disk',15);
    BW = imclose(BW, se);
% 
    se = strel('disk',6);
    BW = imopen(BW, se);
%     BW = imfill(BW, 'holes');
end

function map = colorChangesMap(I)
    redChannel = I(:,:,1);
    greenChannel = I(:,:,2);
    blueChannel = I(:,:,3);
    
    map = (redChannel < 30 & greenChannel < 30 & blueChannel < 30) | ...
        (redChannel > 150 & greenChannel < 125 & blueChannel < 125) | ...
        (redChannel > 240 & greenChannel > 240 & blueChannel > 240);
    %map = imcomplement(map);
end