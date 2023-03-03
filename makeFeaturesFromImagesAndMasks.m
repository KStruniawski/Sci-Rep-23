% Calculates features values based on images and masks, input images and masks paths
% and outputDataFile with will contain all features values that is saved in
% data folder. Function also returns calculated features.

function makeFeaturesFromImagesAndMasks(imagesPath, masksPath, outputDataFile)
    %load column names
    resultsColDesc = load("data/mat/FeaturesColNames.mat").x;
    
    %calculate features
    [resultsNames, resultsClasses, results] = calculateFeatures(length(resultsColDesc)-1,imagesPath,masksPath);
    
    %format results
    resultsNames = split(resultsNames, ".png");
    resultsClasses = split(resultsClasses, ".png");
    resultsNames = resultsNames(~cellfun('isempty',resultsNames));
    resultsClasses = resultsClasses(~cellfun('isempty',resultsClasses));
    
    results(~any(results,2),:) = [];
    
    %normalization -1 to 1
    for i = 2:size(results,2)
        results(:,i) = 2 * mat2gray(results(:,i)) - 1;
    end
    
    results = num2cell(results);
    results = [resultsClasses, results];
    
    T = cell2table(results, 'VariableNames', resultsColDesc, 'RowNames', resultsNames);
    
    %% output files 
    outputDestinationCSV = "data/csv/" + outputDataFile + ".csv";
    outputDestinationMAT = "data/mat/" + outputDataFile + ".mat";
    writetable(T, outputDestinationCSV);
    save(outputDestinationMAT, 'T', '-v7');
    
    %print generated features names
    fprintf("\nFrom %i images generated %i image features: ", length(resultsColDesc));
    resultsColDesc
end

function [resultsNames, resultsClasses, results] = calculateFeatures(numFeatures,imagesPath,masksPath)
    %% files operations
    %go to masks and get list of files
    cd(masksPath);
    imagesMasks = dir;
    cd ..;
    cd ..;

    %go to original and get list of files
    cd(imagesPath);
    imagesOriginal = dir;
    cd ..;
    cd ..;

    
    if length(imagesOriginal) ~= length(imagesMasks)
        ME = MException('MyComponent:noSuchVariable', 'Number of input images and their masks are different!');
        throw(ME)
    end
    
    %% starting variables
    numOfAdditional = numFeatures;
    results = zeros(length(imagesMasks), numOfAdditional);
    resultsNames = [];
    resultsClasses = [];
  
    fprintf('\nCalculating image features values ');
     
    parfor i = 1:length(imagesMasks)
        try
            %get mask image
            maskPath = imagesMasks(i).folder + "/" + imagesMasks(i).name;
            IMask = imread(maskPath);
            
            %get original image
            orgPath = imagesOriginal(i).folder + "/" + imagesOriginal(i).name;
            IOrg = imread(orgPath);
        catch
            continue;
        end
        
        %string formatting
        name = imagesOriginal(i).name;
        class = regexprep(name, '[\d]', '');
        class = erase(class,"_");
        resultsRow = [];
        
        IOrg = filterOriginalImage(IOrg);
       
%         %% Color Features (I1I2I3)
        I1I2I3 = rgb2i1i2i3(IOrg);
        I1 = I1I2I3(:,:,1);
        I2 = I1I2I3(:,:,2);
        I3 = I1I2I3(:,:,3);
        [result] = featuresFromColorChannelHistogram(I1, IMask);
        resultsRow = [resultsRow, result];
        [result] = featuresFromColorChannelHistogram(I2, IMask);
        resultsRow = [resultsRow, result];
        [result] = featuresFromColorChannelHistogram(I3, IMask);
        resultsRow = [resultsRow, result];

        %% xyz alternative features
        [result] = momentsFeaturesXYZ(IOrg, IMask);
        resultsRow = [resultsRow, result];
        
        %% GLCM Features
        [result] = GLCMFeatures(IOrg, IMask);
        for j = 1:size(result,1)
            resultsRow = [resultsRow, result(j,:)];
        end
         
        %% GLRLM Features
        [result] = GLRLMFeatures(IOrg);
        for j = 1:size(result,1)
            resultsRow = [resultsRow, result(j,:)];
        end
        
        %% ADD ALL DATA TO RESULTS
        results(i,:) = resultsRow;
        resultsNames = [resultsNames, name];
        resultsClasses = [resultsClasses, class];
    end
end

%% FEATURES FOR SELECTED CHANNEL AND COLOR SPACE
function [x] = featuresFromColorChannelHistogram(I, IMask)
    I_mean = mean(I(IMask));
    I_std = std(I(IMask));
    
    %histogram based
    I_hist = imhist(I(IMask));
    I_mean_h = mean(I_hist);
    I_var_h = var(I_hist);
    I_kurt = kurtosis(I_hist);
    I_skew = skewness(I_hist);
    I_entr = entropy(I_hist);
    I_e = sum(I(IMask), 'all');
    
    x = [I_mean, I_std, I_mean_h, I_var_h, I_kurt, I_skew, I_entr, I_e];
end

% function [x] = featuresFromColorChannelStatistics(I, IMask)
%     Im = I(IMask);
%     I_mean = mean(Im);
%     I_std = std(Im);
%     I_me = median(Im);
%     I_ad = mean(abs(Im - I_mean)); %avarage deviation
%     I_q1 = quantile(Im,0.25);
%     I_q3 = quantile(Im,0.75);
%     I_moment_1 = sum([mean(Im.^2),mean(Im.^3),mean(Im.^4),mean(Im.^5)]);%sum of moments of 2nd to 5th degree
%     I_moment_2 = sum([mean(Im.^6),mean(Im.^7),mean(Im.^8),mean(Im.^9)]);%sum of moments of 6th to 10th degree
%     
%     x = [I_mean, I_std, I_me, I_ad, I_q1, I_q3, I_moment_1, I_moment_2];
% end

%% calculate moments features in xyz color space
function [x] = momentsFeaturesXYZ(I, IMask)
    I = rgb2xyz(I); 
    x = [];
    for i = 1:3
        Ii = I(:,:,i);
        Im = Ii(IMask);
        I_mean = mean(Im);
        I_std = std(Im);
        I_kurt = kurtosis(Im);
        I_skew = skewness(Im);
        x = [x, I_mean, I_std, I_skew, I_kurt];
    end
end


% function [x] = centroidsCalculations(I, k)
%     Ik = im2uint8(I); 
%     idx_map = imsegkmeans(Ik,k);
%     x = zeros(1, k);
%     for i = 1:k
%         x(i) = mean(I(idx_map==i));
%     end
% end

%% filter original image
function I = filterOriginalImage(I)
    %adaptive histeq
    LAB = rgb2lab(I);
    L = LAB(:,:,1)/100;
    L = adapthisteq(L);
    LAB(:,:,1) = L*100;
    I = lab2rgb(LAB);
    %
    I = imgaussfilt(I);
    I = im2double(I);
end

%% convert rgb image to i1i2i3 color space
function i1i2i3 = rgb2i1i2i3(rgb)
 tm = [1/3 1/3 1/3; 0.5 0 -0.5; -0.25 0.5 -0.25];
 [w, h, c] = size(rgb);
 rgb = reshape(im2double(rgb), [], c)';
 i1i2i3 = (reshape((tm*rgb)', w, h, c)+1)/2;
end