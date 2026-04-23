% 1. Loads image 01 from the folder
img = imread('Assignment_Input/IMG_01.jpg');
grayImg = rgb2gray(img);

% 2. Thresholding using a high threshold of 0.85 to target the bright logo
bw = imbinarize(grayImg, 0.85);

% 3. Morphological cleaning making the swan complete without holes
se = strel('disk', 3);
bw = imclose(bw, se);
bw = imfill(bw, 'holes');

% 4. Area filtering by removing small text and the large background block
stats = regionprops(bw, 'Area', 'PixelIdxList');
areas = [stats.Area];

% Area boundaries keeping objects larger than text but smaller than the sky
swanIdx = find(areas > 5000 & areas < 40000);
bw_final = false(size(bw));
for i = 1:length(swanIdx)
    bw_final(stats(swanIdx(i)).PixelIdxList) = true;
end

% 5. Displays result of the segmentation
figure;
subplot(1, 2, 1); imshow(img); title('Original Input Image');
subplot(1, 2, 2); imshow(bw_final); title('Task 1: Final Segmented Binary Mask');