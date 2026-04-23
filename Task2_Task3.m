%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Task 2. Robust Swan Recognition
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% 1. File setup for images
inputDir = 'Assignment_Input/';
gtDir = 'Assignment_GT/';
outputDir = 'output/';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

inputFiles = dir(fullfile(inputDir, 'IMG_*.JPG'));
numImages = length(inputFiles);
diceScores = zeros(numImages, 1);

% Stores image names for the table
imageNames = cell(numImages, 1); 

fprintf('Found %d images. Starting pipeline\n', numImages);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');

% Loads and preprocesses the images
for k = 1:numImages
    imgName = inputFiles(k).name;
    imageNames{k} = imgName;
    try
        img = imread(fullfile(inputDir, imgName));
    catch
        continue;
    end
    
    [rows, cols, ~] = size(img);
    imgArea = rows * cols;
    imgCenter = [cols/2, rows/2];
    maxDist = sqrt(cols^2 + rows^2)/2;
    
    grayImg = rgb2gray(img);
    hsvImg = rgb2hsv(img);
    
    % ~~~ Stage 1 - Feature extraction ~~~

    % 1. MSER detects stable regions regardless of the contrast
    try
        [regions, ~] = detectMSERFeatures(grayImg, ...
            'RegionAreaRange', [round(imgArea*0.00005), round(imgArea*0.05)], ...
            'ThresholdDelta', 1.0);
        
        mserMask = false(rows, cols);
        % Plots MSERs into a mask
        for i = 1:length(regions)
            pixelList = regions(i).PixelList;
            x = min(max(pixelList(:,1), 1), cols);
            y = min(max(pixelList(:,2), 1), rows);
            idx = sub2ind([rows, cols], y, x);
            mserMask(idx) = true;
        end
    catch
        % Fallback if MSER fails
        mserMask = false(rows, cols);
    end
    
    % 2. Edge density map finds busy vs smooth areas
    [Gmag, ~] = imgradient(grayImg);
    edgeMap = Gmag > (max(Gmag(:)) * 0.05);
    edgeBlob = imclose(edgeMap, strel('rectangle', [3, round(cols*0.05)]));
    edgeBlob = imclose(edgeBlob, strel('rectangle', [round(rows*0.02), 3]));
    
    % 3. Color saliency high value and low saturation means white logo
    S = hsvImg(:,:,2);
    V = hsvImg(:,:,3);
    whiteMask = (S < 0.45) & (V > 0.6);
    
    % 4. Dark saliency for any inverted logo's
    darkMask = grayImg < 60;
    
    % ~~~ Stage 2 - Candidate generation ~~~
    candidates = {};
    
    % C1. MSER blob dilates MSER to connect the letters into words
    candidates{end+1} = imclose(mserMask, strel('disk', 6));
    
    % C2. Edge blob for texture regions
    candidates{end+1} = imfill(edgeBlob, 'holes');
    
    % C3. White objects for standard logo's
    candidates{end+1} = imclose(whiteMask, strel('disk', 4));
    
    % C4. Dark objects for inverted logo's
    candidates{end+1} = imclose(darkMask, strel('disk', 4));
    
    % C5. Adaptive threshold as the backup
    candidates{end+1} = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.5);
    
    % C6: Gray background detection fo bright objects on gray backgrounds
    meanS = mean(S(:));
    meanV = mean(V(:));
    
    % Check if the background has a low saturation and medium value
    if meanS < 0.25 && meanV > 0.45 && meanV < 0.75
        grayBgMask = (S < 0.20) & (V > 0.50) & (V < 0.80);
        brightLogo = (V > 0.65) & (S < 0.30);
        grayCandidate = brightLogo & ~grayBgMask;
        candidates{end+1} = imclose(grayCandidate, strel('disk', 5));
    end
    
    % C7. Fine edge detection for smaller distant logo's
    fineEdges = edge(grayImg, 'Canny', [0.04, 0.12]);
    fineEdges = imclose(fineEdges, strel('disk', 3));
    fineEdges = imfill(fineEdges, 'holes');
    fineEdges = bwareaopen(fineEdges, round(imgArea*0.00003));
    candidates{end+1} = fineEdges;
    
    % ~~~ Stage 3 - Tournament winner selection ~~~

    bestMask = false(rows, cols);
    bestScore = -inf;
    
    for c = 1:length(candidates)
        mask = candidates{c};
        
        % Pre-filter noise being relaxed for smaller logo's
        mask = bwareaopen(mask, round(imgArea*0.0002));
        
        % Connect components through fusing the text together 
        mask = imclose(mask, strel('rectangle', [5, round(cols*0.04)]));
        mask = imfill(mask, 'holes');
        
        if sum(mask(:)) == 0, continue; end
        
        stats = regionprops(mask, 'Area', 'BoundingBox', 'Centroid', 'Solidity', 'PixelIdxList', 'Extent');
        
        for i = 1:length(stats)
            
            % ~~~ Logic for image rejection ~~~
            
            % 1. Size constraints being relaxed for smaller logo's
            relArea = stats(i).Area / imgArea;
            if relArea < 0.0001 || relArea > 0.60
                continue;
            end
            
            % 2. Aspect ratio boundaries as the logo is more wide than tall
            bb = stats(i).BoundingBox;
            aspect = bb(3) / bb(4);
            if aspect < 0.15 || aspect > 20
                continue; 
            end
            
            % 3. Border touch penalisation, but not strictly rejected
            if bb(2) < 5 || (bb(2)+bb(4)) > rows-5
                borderPenalty = 0.5;
            else
                borderPenalty = 0;
            end
            
            % ~~~ Scoring logic ~~~
            
            % Metric A: Centrality due to the central nature of the logo's
            dist = sqrt(sum((stats(i).Centroid - imgCenter).^2));
            scoreDist = 1 - (dist / maxDist);
            
            % Metric B: Internal detail looking at the standard deviation of pixel intensities inside the box
            pixelVals = double(grayImg(stats(i).PixelIdxList));
            localStd = std(pixelVals);
            scoreDetail = min(localStd / 60, 1.0);
            
            % Metric C: Solidity, questioning if its a coherant box
            scoreSolid = stats(i).Solidity;
            
            % Metric D: Size Optimality being between 5% and 15% of the image
            scoreSize = 1 - min(abs(relArea - 0.10) * 4, 1);
            
            % Metric E: Contrast sampling a ring around the object
            thisObj = false(rows, cols);
            thisObj(stats(i).PixelIdxList) = true;
            ring = imdilate(thisObj, strel('disk', 10)) & ~thisObj;
            if sum(ring(:)) > 0
                bgMean = mean(grayImg(ring));
                objMean = mean(grayImg(thisObj));
                scoreContrast = abs(objMean - bgMean) / 255;
            else
                scoreContrast = 0;
            end
            
            % Weighted score through detail and contrast being the most important discriminators
            totalScore = (scoreDist * 0.20) + ...
                         (scoreDetail * 0.30) + ...
                         (scoreContrast * 0.25) + ...
                         (scoreSolid * 0.15) + ...
                         (scoreSize * 0.10) - ...
                         borderPenalty;
            
            if totalScore > bestScore
                bestScore = totalScore;
                
                % ~~~ Refinement Strategy ~~~
                % We have the winning bounding box/blob to generate the final mask
                
                % Gets the bounding box image
                r1 = max(1, floor(bb(2))); r2 = min(rows, floor(bb(2)+bb(4)));
                c1 = max(1, floor(bb(1))); c2 = min(cols, floor(bb(1)+bb(3)));
                cropImg = grayImg(r1:r2, c1:c2);
                
                % Adaptive threshold inside of the box extracting the letters from the background block
                localMask = imbinarize(cropImg, 'adaptive', 'Sensitivity', 0.55);
                
                % Polarity checking if the box mean is bright, assuming dark text
                if mean(cropImg(:)) > 140
                    localMask = ~localMask;
                end
                
                % Places back into full size mask
                finalMask = false(rows, cols);
                finalMask(r1:r2, c1:c2) = localMask;
                
                % 5. Combining with blob using intersection & fill to get solid letters
                thisBlob = false(rows, cols);
                thisBlob(stats(i).PixelIdxList) = true;
                
                bestMask = finalMask & thisBlob;
                bestMask = imfill(bestMask, 'holes');
                
                % If adaptive thresholding killed the object it reverts to the solid blob
                if sum(bestMask(:)) < (stats(i).Area * 0.15)
                    bestMask = thisBlob;
                end
            end
        end
    end
    
    % ~~~ Isolating the swan from the text in the block ~~~

    % The detection finds the whole logo block
    predMask = isolateSwanOnly(bestMask, grayImg);
    
    % Fallback for any failed detections
    [rows, cols] = size(grayImg);
    imgArea = rows * cols;
    
    needsFallback = false;
    
    if sum(predMask(:)) == 0
        needsFallback = true;
    elseif sum(predMask(:)) > imgArea * 0.12
        needsFallback = true;
    else
        % Check aspect ratio and position, if its extremely wide or off-center, its likely wrong
        stats = regionprops(predMask, 'BoundingBox', 'Centroid');
        if ~isempty(stats)
            bb = stats(1).BoundingBox;
            aspect = bb(3) / bb(4);
            
            % Checks the centrality
            imgCenter = [cols/2, rows/2];
            dist = sqrt(sum((stats(1).Centroid - imgCenter).^2));
            maxDist = sqrt(cols^2 + rows^2) / 2;
            centrality = 1 - (dist / maxDist);
            
            % Triggers if very wide or very off-center, also if the image is very tiny
            if aspect > 4.0 || centrality < 0.35 || sum(predMask(:)) < imgArea * 0.0002
                needsFallback = true;
            end
        end
    end
    
    if needsFallback
        predMask = fallbackSmallCentered(img, grayImg);
    end
    
    % Final polishing
    if sum(predMask(:)) > 0
        predMask = bwareaopen(predMask, round(sum(predMask(:))*0.02));
    end

    % ~~~ Task 3. Evaluation ~~~
    gtName = strrep(imgName, '.JPG', '_GT.JPG');
    try
        targetGT = imread(fullfile(gtDir, gtName));
        % Handles cases where the GT might be RGB or not boolean
        if size(targetGT, 3) > 1, targetGT = rgb2gray(targetGT); end
        targetGT = targetGT > 127;
        
        % Computes the dice score using the formula: DS = 2*|A ∩ B| / (|A| + |B|)
        intersection = sum(predMask & targetGT, 'all');
        totalPixels = sum(predMask, 'all') + sum(targetGT, 'all');
        
        if totalPixels > 0
            diceScores(k) = (2 * intersection) / totalPixels;
        else
            % If both are empty, score 1, if one is empty, score 0
            if sum(targetGT(:)) == 0 && sum(predMask(:)) == 0
                diceScores(k) = 1.0;
            else
                diceScores(k) = 0.0;
            end
        end
        
        fprintf('Pair %d [%s]: Dice Score = %.4f\n', k, imgName, diceScores(k));
        
        % Saves the result of the image
        imwrite(uint8(predMask)*255, fullfile(outputDir, ['Result_', imgName]));
        
    catch ME
        fprintf('Warning: GT not found or error for %s (%s)\n', imgName, ME.message);
        diceScores(k) = 0;
    end
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Task 3. Perfomance Evaluation
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fprintf('\n');
fprintf(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf(' Task 3. Perfomance Evaluation\n');
fprintf(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% Calculates the statistics
validScores = diceScores(diceScores >= 0); % Include 0s in average
meanDice = mean(validScores);
stdDice = std(validScores);

% Displays the formula
fprintf('Dice Score Formula:\n');
fprintf('DS = 2 * |A ∩ B| / (|A| + |B|)\n');
fprintf('where A = predicted mask, B = ground truth mask\n');
fprintf('DS ranges from 0 = no overlap, to 1 = perfect match\n\n');

% Creates and displays the results table
fprintf('%-15s | %-12s\n', 'Image', 'Dice Score');
fprintf('--------------------------------------------------\n');
for k = 1:numImages
    fprintf('%-15s | %.4f\n', imageNames{k}, diceScores(k));
end
fprintf('--------------------------------------------------\n');
fprintf('%-15s | %.4f\n', 'Mean', meanDice);
fprintf('%-15s | %.4f\n', 'Std Dev', stdDice);
fprintf('--------------------------------------------------\n\n');

% Creates a formatted table for easier viewing
resultsTable = table(imageNames, diceScores, 'VariableNames', {'Image', 'DiceScore'});

% Adds a summary row
summaryTable = table({'Mean'; 'Std Dev'}, [meanDice; stdDice], ...
    'VariableNames', {'Statistic', 'Value'});

% Displays the table
fprintf('MATLAB Table Format (for report):\n\n');
disp(resultsTable);
fprintf('\nSummary Statistics:\n');
disp(summaryTable);

% Saves the results to CSV file
resultsFile = fullfile(outputDir, 'dice_scores.csv');
writetable(resultsTable, resultsFile);
fprintf('\nResults saved to: %s\n', resultsFile);

% Creates a visualisation of the results
figure('Name', 'Task 3: Dice Score Performance', 'Position', [100, 100, 1000, 600]);

% Generates a bar chart of dice scores
subplot(1, 2, 1);
bar(diceScores, 'FaceColor', [0.2 0.4 0.7]);
hold on;
yline(meanDice, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean = %.4f', meanDice));
yline(meanDice + stdDice, 'g:', 'LineWidth', 1.5, 'Label', '+1 Std');
yline(meanDice - stdDice, 'g:', 'LineWidth', 1.5, 'Label', '-1 Std');
xlabel('Image Index');
ylabel('Dice Score');
title('Dice Score per Image');
grid on;
ylim([0, 1]);
set(gca, 'XTick', 1:numImages);
xtickangle(45);

% Distribution histogram creation
subplot(1, 2, 2);
histogram(diceScores, 10, 'FaceColor', [0.2 0.4 0.7], 'EdgeColor', 'black');
xlabel('Dice Score');
ylabel('Frequency');
title('Distribution of Dice Scores');
grid on;
text(0.05, max(ylim)*0.9, sprintf('Mean = %.4f\nStd = %.4f', meanDice, stdDice), ...
    'FontSize', 10, 'BackgroundColor', 'white');

% Saves the figure
saveas(gcf, fullfile(outputDir, 'dice_score_analysis.png'));
fprintf('Performance visualization saved to: %s\n', fullfile(outputDir, 'dice_score_analysis.png'));

fprintf(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');
fprintf(' Task 3 Complete - All results saved to %s\n', outputDir);
fprintf(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');

% Function to isolate the swan from the logo block

function swanMask = isolateSwanOnly(logoBlock, grayImg)
    % The detected block contains swan + text
    % We need to extract ONLY the swan
    
    [rows, cols] = size(grayImg);
    
    if sum(logoBlock(:)) == 0
        swanMask = logoBlock;
        return;
    end
    
    % Strategy: Split vertically and take upper portion
    % Swan is almost always in the top 60% of the logo block
    
    stats = regionprops(logoBlock, 'BoundingBox');
    if isempty(stats)
        swanMask = logoBlock;
        return;
    end
    
    bb = stats(1).BoundingBox;
    
    % Gets the top 55% of the logo the block 
    topCutoff = bb(2) + bb(4) * 0.55;
    
    % Creates a mask for upper region
    upperMask = false(rows, cols);
    upperMask(1:round(topCutoff), :) = true;
    
    % Extracts the swan region
    swanRegion = logoBlock & upperMask;
    
    % Cleans up, removing small text fragments, keeping the largest component
    swanRegion = bwareaopen(swanRegion, round(sum(logoBlock(:)) * 0.02));
    
    cc = bwconncomp(swanRegion);
    if cc.NumObjects > 0
        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, idx] = max(numPixels);
        swanMask = false(rows, cols);
        swanMask(cc.PixelIdxList{idx}) = true;
    else
        swanMask = swanRegion;
    end
    
    swanMask = imfill(swanMask, 'holes');
    
    % Fallback for if we lost too much, returns to the original
    if sum(swanMask(:)) < sum(logoBlock(:)) * 0.05
        swanMask = logoBlock;
    end
end

% Function for fallback small centered detection

function mask = fallbackSmallCentered(img, grayImg)
    % For images where main detection fails. Looking for small, highly centered, high contrast white objects
    [rows, cols] = size(grayImg);
    imgArea = rows * cols;
    imgCenter = [cols/2, rows/2];
    
    % Converts to HSV for better white detection
    hsvImg = rgb2hsv(img);
    S = hsvImg(:,:,2);
    V = hsvImg(:,:,3);
    
    % Detects white/bright objects with multiple strategies
    mask1 = (S < 0.30) & (V > 0.65);  
    mask2 = (S < 0.45) & (V > 0.62); 
    mask3 = (grayImg > 180);           
    
    % Extra strategy for very small logos
    mask4 = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.6);
    
    candidates = {mask1, mask2, mask3, mask4};
    
    bestMask = false(rows, cols);
    bestScore = -inf;
    
    for c = 1:length(candidates)
        cmask = candidates{c};
        cmask = imclose(cmask, strel('disk', 2));
        cmask = imfill(cmask, 'holes');
        cmask = bwareaopen(cmask, round(imgArea * 0.00005));
        
        cc = bwconncomp(cmask);
        stats = regionprops(cc, 'Area', 'Centroid', 'BoundingBox', 'PixelIdxList', 'Solidity');
        
        for i = 1:length(stats)
            relArea = stats(i).Area / imgArea;
            
            % Focuses on very small to medium objects
            if relArea < 0.00005 || relArea > 0.12
                continue;
            end
            
            bb = stats(i).BoundingBox;
            aspect = bb(3) / bb(4);
            
            % Swan like aspect ratio, being relaxed for smaller logo's
            if aspect < 0.3 || aspect > 3.5
                continue;
            end
            
            % Calculates centrality, having to be reasonably centered
            dist = sqrt(sum((stats(i).Centroid - imgCenter).^2));
            maxDist = sqrt(cols^2 + rows^2) / 2;
            scoreCentral = 1 - (dist / maxDist);
            
            % Skips if it's too far from center
            if scoreCentral < 0.25
                continue;
            end
            
            % Prefer square shapes
            scoreAspect = 1 - abs(aspect - 1.0);
            
            % Checks the contrast
            tempMask = false(rows, cols);
            tempMask(stats(i).PixelIdxList) = true;
            ring = imdilate(tempMask, strel('disk', 12)) & ~tempMask;
            
            if sum(ring(:)) > 50
                bgMean = mean(grayImg(ring));
                objMean = mean(grayImg(tempMask));
                scoreContrast = abs(objMean - bgMean) / 255;
            else
                scoreContrast = 0;
            end
            
            scoreSolid = stats(i).Solidity;
            % Size preference of medium sizes
            scoreSize = 1 - abs(log10(max(relArea, 0.0001)) - log10(0.02));
            scoreSize = max(0, min(1, scoreSize));
            
            % Heavily weighted centrality, contrast, and size for small logos
            totalScore = scoreCentral * 0.35 + ...
                         scoreContrast * 0.30 + ...
                         scoreAspect * 0.15 + ...
                         scoreSolid * 0.10 + ...
                         scoreSize * 0.10;
            
            if totalScore > bestScore
                bestScore = totalScore;
                bestMask = false(rows, cols);
                bestMask(stats(i).PixelIdxList) = true;
            end
        end
    end
    
    mask = imfill(bestMask, 'holes');
end