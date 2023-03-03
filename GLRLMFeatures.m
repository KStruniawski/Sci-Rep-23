% Calculate GLRLM features for original image in grayscale
function results = GLRLMFeatures(I)
    % convert image to grayscale
    I = rgb2gray(I);

    % calculate glrlm matrix
    [GLRLMS, ~]= grayrlmatrix(I);
    
    % calculate glrlm matrix properties
    stats = grayrlprops(GLRLMS);
    results = mean(stats,1)';
end