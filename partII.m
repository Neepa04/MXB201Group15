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

S = U(:, 1:47);       % Eigenface space of largest singular values

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

if layout^2 < moustache_faces_cols
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
uc_vectors = c_vectors(:, mask2);

% Isolate columns of unique faces matrix corresponding to faces with a moustache
moustache_level2 = 2000;                                 % Moustache level

mask3 = uc_vectors(13,:) >= moustache_level2;
uf_moustache_faces = uniquefaces(:,mask3);
uf_moustache_faces_cols = size(uf_moustache_faces, 2);   % Number of detected moustache faces (columns)   


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
uf_moustache_faces_vis = reshape(uf_moustache_faces, rows, cols, uf_moustache_faces_cols);

% Producing an approximate square tiled layout for any moustache level
layout = round(sqrt(uf_moustache_faces_cols));

if layout^2 < uf_moustache_faces_cols
    layout2 = layout + 1;
else
    layout2 = layout;
end

figure
tiledlayout(layout, layout2, "Padding","Compact")
sgtitle('Unique Faces: Faces Detected Visualisation', 'FontWeight', 'Bold')

for i = 1: uf_moustache_faces_cols
    nexttile
    imagesc(uf_moustache_faces_vis(:,:, i))
    colormap('gray')
    axis off
end

fprintf("There are %d unique faces detected with a moustache (Moustache Level = %d).\n", ...
    uf_moustache_faces_cols, moustache_level2)

% There are two faces with a moustache from the sample of unique faces. From a moustache level standpoint of 1800, 
% the detector successfully detected 2/2 images of an individual with a moustache giving a perfect accuracy (100%)
% Optimal moustache level: < 1672.127