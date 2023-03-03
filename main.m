% main function to run makingMasks, featuresCalculations and classification

% function main()
%     % setup external libraries (only if error occurs)
%     cd resources/fspackage;
%     load_fspackage;
%     % path to the weka.jar (external library for feature selection)
%     javaaddpath('/Applications/R2019b.app/java/jar/weka.jar');
    
    % paths of input images and their masks
%     imagePath = "in/original";
%     maskPath = "out/original";
%     featuresFileName = "FeaturesSet";
%     
% %     % first calculations of masks and features
%     makeMasks2(imagePath, maskPath);
%     makeFeaturesFromImagesAndMasks(imagePath,maskPath,featuresFileName);
%     makeClassification();
    
    % loading calculated features for classification     
%     
    res1 = makeClassification("FeaturesSet3", 'infogain');
%     res5 = makeClassification("FeaturesSet3", 'mrmr');
    beep;
%     res8 = makeClassification("FeaturesSet2", 'sbmlr');
%     res9 = makeClassification("FeaturesSet3", 'sbmlr');
% end
