% MXB201 Project Part II.
%% Initialisation
clear
d = dir('faces/*.pgm');
N = length(d);
I = imread([d(1).folder, '/', d(1).name]);
[rows,cols] = size(I);
M = rows*cols;
A = zeros(M, N);  % big matrix, whose columns are the images

%% Read images as columns of the matrix
for j = 1:N
    I = imread([d(j).folder, '/', d(j).name]);
    A(:,j) = I(:);
end

%% Calculate and visualise mean face

% Calculating mean face
mean_face = sum(A,2) / size(A, 2);

% Visualising mean face
figure
mean_face_vis = reshape(uint8(mean_face), rows, cols);
imshow(mean_face_vis, 'InitialMagnification', 'Fit')
title('Mean Face', 'FontWeight', 'Bold')

%% Calculate mean-centred SVD

% Calculating mean-centred matrix
mean_centred = A - mean_face;

% Mean-centred SVD
[U, Sigma, V] = svd(mean_centred, 'econ');
Sigma_values = diag(Sigma);

%% Visualise first 20 eigenfaces

% Reshaping U for the eigenfaces of A
eigenfaces = reshape(U, rows, cols, N);

% Visualising first 20 eigenfaces
figure
tiledlayout(4, 5, 'Padding','Compact')
sgtitle('First 20 Eigenfaces', 'FontWeight', 'Bold')

for x = 1:20
    nexttile
    imagesc(eigenfaces(:,:,x));
    colormap('gray')
    title(num2str(x))
    axis off
end

%% Calculate coordinate vectors

% Since the singular values deplete by a magnitude of 1 from the 47th to
% the 48th singular value, the orthogonal matrix of U is truncated to 47
% columns.

% Truncating U with the most significant singular values
S = U(:, 1:47);       % Eigenface space of largest singular values

% Visualising singular values 
SV = 47;              % Chosen singular value

figure
hold on
plot(Sigma_values, '.', 'MarkerSize', 10)
plot(SV, Sigma_values(SV), 'r.', 'MarkerSize', 20);
title('Singular Values of Mean-Centred SVD')
xlabel('nth Singular Value'), ylabel('Singular Value Magnitude')
xlim([1 450])

text(SV + 10, Sigma_values(SV) + 5000, sprintf('[%d, %.2f]', SV, Sigma_values(SV)), ...
    'Color', [0.5 0 0], 'FontSize', 10)


% Coordinate vectors (Least Squares Solution)
c_vectors = S'*A;

%% Demonstrate rudimentary moustache detector

% The 13th column of the eigenface space corresponds to the moustache characteristic
% Therefore, the 13th row of the coordinate vector determines the **moustache level** apparent in the photo
% Below moustache level = moustache undetected,   Equal or above moustache level = moustache detected

% The sample will be the entire face database given (all columns of A i.e. 1000 photos)

% Isolate columns of A corresponding to faces with a moustache
moustache_level = 2000;                             % Moustache Level

mask = c_vectors(13,:) >= moustache_level;      
moustache_faces = A(:, mask);
moustache_faces_cols = size(moustache_faces, 2);    % Number of moustache faces (columns) detected


% Visualising faces with detected moustache
moustache_faces_vis = reshape(moustache_faces, rows, cols, moustache_faces_cols);

% Producing an approximate square tiled layout for any moustache level
layout = round(sqrt(moustache_faces_cols));

if layout == 0
    disp("Zero faces detected for all faces sample. Moustache level is too high.")
    return
elseif layout^2 < moustache_faces_cols
    layout2 = layout + 1;
else
    layout2 = layout;
end

figure
colnum_moustacheface = find(mask);          % Column numbers of moustache detected from A
tiledlayout(layout, layout2, 'Padding', 'Compact', 'TileSpacing', 'Compact')
sgtitle('All Faces: Faces Detected Visualisation', 'FontWeight', 'Bold')

for i = 1:moustache_faces_cols
    nexttile
    imagesc(moustache_faces_vis(:,:, i))
    colormap('gray')
    title(['Photo ' num2str(colnum_moustacheface(i))]);
    axis off
end

fprintf("There are %d faces detected with a moustache (Moustache Level = %d).\n", ...
    moustache_faces_cols, moustache_level)

% From a moustache level standpoint of 1900, the detector successfully detected 29/30 images of an individual with a
% moustache giving an accuracy of 97%


%% ALTERNATIVE: SMALLER SAMPLE

% Moustache detector on a sample of 35 unique faces

% Matrix of unique faces only
mask2 = 1:29:1000;
uniquefaces = A(:,mask2);      

% Coordinate vectors of each unique face
c_vectors2 = c_vectors(:, mask2);

% Isolate columns of unique faces matrix corresponding to faces with a moustache
moustache_level2 = 1847.3;                                 % Moustache level
max_level = max(c_vectors2(13,:));                       % Maximum level applicable

mask3 = c_vectors2(13,:) >= moustache_level2;
moustache_faces2 = uniquefaces(:,mask3);
moustache_faces_cols2 = size(moustache_faces2, 2);       % Number of detected moustache faces (columns)   


% Visualising unique faces with a moustache (Yes or No)
uniquefaces_vis = reshape(uniquefaces, rows, cols, size(uniquefaces, 2));

figure
tiledlayout(5, 7, 'Padding', 'Compact', 'TileSpacing', 'Compact')
sgtitle('Unique Faces: Moustache Detector Results (Yes or No)', 'FontWeight', 'Bold')

for u = 1:size(uniquefaces,2)
    nexttile
    imagesc(uniquefaces_vis(:,:, u))
    colormap('gray')
    if mask3(u) == 1
        title('Yes', 'Color', [0 0.5 0])
    else
        title('No', 'Color', [0.5 0 0])
    end
    axis off
end


% Visualising unique faces with detected moustache
moustache_faces_vis2 = reshape(moustache_faces2, rows, cols, moustache_faces_cols2);

% Producing an approximate square tiled layout for any moustache level
layout = round(sqrt(moustache_faces_cols2));

if layout == 0
    fprintfs("Zero faces detected for unqiue faces sample." + ...
        " Moustache level is too high ( > %.2f )./n", max_level)
    return
elseif layout^2 < moustache_faces_cols
    layout2 = layout + 1;
else
    layout2 = layout;
end

figure
tiledlayout(layout, layout2, "Padding","Compact")
sgtitle('Unique Faces: Faces Detected Visualisation', 'FontWeight', 'Bold')

for i = 1: moustache_faces_cols2
    nexttile
    imagesc(moustache_faces_vis2(:,:, i))
    colormap('gray')
    axis off
end

fprintf("There are %d unique faces detected with a moustache (Moustache Level = %d).\n", ...
    moustache_faces_cols2, moustache_level2)


% ACCURACY %

% The accuracy of the detector is determined by using True Positive (TP), True Negative (TN),
% False Positive (FP), and False Negative (FN) values into the accuracy formula:

% Accuracy = TP + TN / (TP + TN + FP + FN).     (EvidentlyAI, 2025)

% There are 2 faces with a moustache in the sample of 35 unique faces
% General TP, TN, FP and FN values for any moustache level

% Iniitialising true moustache position
groundtruth = [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];  
groundtruth = logical(groundtruth);

TP = sum((groundtruth == 1) & (mask3 == 1));
TN = sum((groundtruth == 0) & (mask3 == 0)); 
FP = sum((groundtruth == 0) & (mask3 == 1));
FN = sum((groundtruth == 1) & (mask3 == 0));

% Thus, 
Accuracy = (TP + TN) / (TP + TN + FP + FN);
fprintf("Unique Faces Sample Accuracy: %.2f%% \n", Accuracy * 100)

% However, it is important to check the reliability of the computed accuracy as it may be undermined 
% by the imbalance in the sample, where unique faces without a moustache significantly outnumber those
% with a moustache. Since FP and FN = 0, the accuracy calculated is deemed reliable.

% Therefore, from a moustache level standpoint of 1800, the accuracy of the moustache detector on a sample of 
% 35 unique faces is flawless at a 100% accuracy. *HOWEVER*, this accuracy is specific to the sample used, and
% may not generalise to larger or differently composed datasets.


% Optimal moustache level: Accuracy vs. Moustache Level Plot

% Calculating accuracy for moustache levels of 1 to the max level
moustache_level_axis = 1:max_level;

for i = 1:max_level
    mask3 = c_vectors2(13,:) >= i;
    TP = sum((groundtruth == 1) & (mask3 == 1));
    TN = sum((groundtruth == 0) & (mask3 == 0)); 
    FP = sum((groundtruth == 0) & (mask3 == 1));
    FN = sum((groundtruth == 1) & (mask3 == 0));
    Accuracy(i,:) = (TP + TN) / (TP + TN + FP + FN);
end

% Calculating lower bound and setting upper bound of optimal interval
moustache_coord_values = c_vectors2(13, :);        % 1x35 matrix of moustache coordinate values
no_moustache_values = moustache_coord_values;       
no_moustache_values(:, groundtruth) = 0;           % Moustache coordinate values of faces without a moustache
        
% Interval of Optimal Moustache Level
lower_bound = max(no_moustache_values);            % Max moustache coordinate value of faces without a moustache
upper_bound = moustache_coord_values(:, 9);        % Lowest moustache coordinate value of faces with a moustache

figure
plot(moustache_level_axis, Accuracy * 100, 'HandleVisibility', 'off')
title('Accuracy vs. Moustache Level')

% Adding appropriate lines, axis titles and text
hold on
xline(moustache_level2, 'b:', 'DisplayName', ...
    sprintf('Chosen Moustache Level (%.2f)', moustache_level2));
xline(lower_bound, '--r', 'DisplayName',  'Optimal Moustache Level Interval');
xline(upper_bound, '--r', 'HandleVisibility',  'off');
xlabel('Moustache Level'), ylabel('Accuracy (%)')
text(upper_bound + 100, Accuracy(round(upper_bound)) * 100 - 8, ...
    sprintf('100%% at Moustache Level\n ∈ [%.2f, %.2f]', ...
lower_bound, upper_bound), 'Color', [0.5 0 0], 'FontSize', 7.5)

legend('Show', 'Location', 'Northwest', 'FontSize', 8)

xlim([0 max_level]), ylim([30 120])