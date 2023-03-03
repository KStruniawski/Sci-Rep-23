% calculate GLCM features seperately for Red, Green and Blue color channels
% for RGB color space
function results = GLCMFeatures(I, mask)
    results = [];
    % loop throught the 3 different color spaces and add all to the results
    % vector
    for i = 1:3
        results = [results; GLCMFeaturesFromChannel(I(:,:,i), mask)];
    end
end

function results = GLCMFeaturesFromChannel(I, mask)
    % Hadamard multiplication of image with its mask
    I = mask.*I;
    
    %Calculate GLCM matrix for angles 0, 45, 90, 135
    GLCM = graycomatrix(I,'Offset',[0 1; -1 1;-1 0;-1 -1]);
    
    %% Based od GLCM matrix measure statistics
    size_glcm_3 = size(GLCM,3);
    glcm_contrast = zeros(1,size_glcm_3);
    glcm_correlation = glcm_contrast;
    glcm_energy = glcm_contrast;
    glcm_entropy = glcm_contrast;
    glcm_homogeneity = glcm_contrast;
    glcm_autocorrelation = glcm_contrast;
    glcm_dissimilarity = glcm_contrast;
    glcm_cluster_prominence = glcm_contrast;
    glcm_inverse_difference = glcm_contrast;
    
    parfor k = 1:size_glcm_3
        GLCM_k = GLCM(:,:,k);
        GLCM_k = GLCM_k + GLCM_k';
        glcm_sum = sum(GLCM_k,'all');
        GLCM_k = GLCM_k./glcm_sum;
        
        mi_i = 0; mi_j = 0; sig_i = 0; sig_j = 0;
        for i = 1:size(GLCM_k,1)
            for j = 1:size(GLCM_k,1)
                mi_i = mi_i + (i) * GLCM_k(i,j);
                mi_j = mi_j + (j) * GLCM_k(i,j);
            end
        end
        
        for i = 1:size(GLCM_k,1)
            for j = 1:size(GLCM_k,1)
                sig_i = sig_i + (i-sig_i)^2 * GLCM_k(i,j);
                sig_j = sig_j + (j-sig_j)^2 * GLCM_k(i,j);
            end
        end
                
        for i = 1:size(GLCM_k,1)
            for j = 1:size(GLCM_k,1)
                glcm_contrast(k) = glcm_contrast(k) + ((i-j)^2*GLCM_k(i,j));
                glcm_correlation(k) = glcm_correlation(k) + (((i-mi_i)*(j-mi_j)*GLCM_k(i,j))/(sig_i*sig_j));
                glcm_energy(k) = glcm_energy(k) + GLCM_k(i,j)^2;
                glcm_homogeneity(k) = glcm_homogeneity(k) + (GLCM_k(i,j) / (1+(i-j)^2));
                glcm_autocorrelation(k) = glcm_autocorrelation(k) + (i*j*GLCM_k(i,j));
                glcm_dissimilarity(k) = glcm_dissimilarity(k) + (abs(i-j)*GLCM_k(i,j));
                glcm_cluster_prominence(k) = glcm_cluster_prominence(k) + ((i+j-mi_i-mi_j)^4*GLCM_k(i,j));
                glcm_inverse_difference(k) = glcm_inverse_difference(k) + (GLCM_k(i,j) / (1+abs(i-j)));
                
                if GLCM_k(i,j) ~= 0
                    glcm_entropy(k) = glcm_entropy(k) - (log2(GLCM_k(i,j))*GLCM_k(i,j));
                end
            end
        end
    end
    
    results = [mean(glcm_contrast);mean(glcm_correlation);mean(glcm_energy);mean(glcm_entropy)];
    results = [results;mean(glcm_homogeneity);mean(glcm_autocorrelation);mean(glcm_cluster_prominence);mean(glcm_inverse_difference);mean(glcm_dissimilarity)];
end