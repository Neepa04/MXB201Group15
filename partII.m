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
mean_face_vis = reshape(uint8(mean_face), rows, cols);
imshow(mean_face_vis, 'InitialMagnification', 'Fit')
title('Mean Face Visualisation')

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
sgtitle('First 20 Eigenfaces Visualisation')

for x = 1:20
    nexttile
    imagesc(eigenfaces(:,:,x));
    colormap('gray')
    title(num2str(x))
    axis off
end

%% Calculate coordinate vectors

S = U(:, 1:47);                        % Eigenface space of largest singular values

% Coordinate vectors (Least Squares Solution)
c_vectors = zeros(47, N);              % Initialising coordinate vector matrix

for i = 1:N
    LHS = S'*S;
    RHS = S'*A(:, i);
    c_vectors(:,i) = LHS \ RHS;
end

%% TEST
recon_face = S*c_vectors(:,30);

figure
recon_face_vis = reshape(uint8(recon_face), rows, cols);
imshow(recon_face_vis, 'InitialMagnification', 'Fit')
title('TEST: Recon face')

%% Demonstrate rudimentary moustache detector

% The 13th column of the eigenface space corresponds to the moustache characteristic
% Therefore, the 13th row of the coordinate vector determines the **moustache level** apparent in the photo
% Below moustache level = moustache undetected,   Equal or above moustache level = moustache detected


% Isolate columns of A corresponding to faces with a moustache
moustache_level = 2000;        % Moustache Level

mask = c_vectors(13,:) >= moustache_level;      
moustache_faces = A(:,mask);

% Visualising faces with detected moustache
moustache_faces_vis = reshape(moustache_faces,rows, cols, size(moustache_faces, 2));

% Producing an approximate square tiled layout for any moustache level
layout = round(sqrt(size(moustache_faces, 2)));

if layout^2 < size(moustache_faces, 2)
    layout2 = layout + 1;
else
    layout2 = layout;
end

figure
tiledlayout(layout, layout2, 'Padding', 'Compact')
sgtitle('Visualisation of Faces with Detected Moustache')

for i = 1:size(moustache_faces, 2)
    nexttile
    imagesc(moustache_faces_vis(:,:, i))
    colormap('gray')
    axis off
end

%% TEST
        % TEST; Visualisation of selected faces
figure
tiledlayout(6, 6)
sgtitle('TEST: Visualisation of selected faces')

face = A(:,1:36);
face_vis = reshape(face, rows, cols, 36);

for u = 1:36
    nexttile
    imagesc(face_vis(:,:, u))
    colormap('gray')
    axis off
end



        % TEST: Visualisation of each unique face
figure
tiledlayout(5, 7)
sgtitle('TEST: Visualisation of each unique face')

mask2 = 1:29:1000;
faces = A(:,mask2);         % Matrix of unique faces only

faces_vis = reshape(faces, rows, cols, size(faces, 2));

for u = 1:size(faces,2)
    nexttile
    imagesc(faces_vis(:,:, u))
    colormap('gray')
    axis off
end


% Coordinate vectors of each unique face (Least Squares Solution)
uc_vectors = zeros(47, 35);           % Initialising unique coordinate vector matrix

for i = 1:35
    LHS = S'*S;
    RHS = S'*faces(:, i);
    uc_vectors(:,i) = LHS \ RHS;
end

% Isolate columns of unique faces matrix corresponding to faces with a moustache
mask3 = uc_vectors(13,:) > 2000;
moustache_faces2 = faces(:,mask3);

% Visualising unique faces with detected moustache
moustache_faces_vis2 = reshape(moustache_faces2,rows, cols, size(moustache_faces2, 2));

figure
tiledlayout(1, size(moustache_faces2, 2), "Padding","Compact")
sgtitle('TEST: Visualisation of Unique Faces with Detected Moustache')

for i = 1:size(moustache_faces2, 2)
    nexttile
    imagesc(moustache_faces_vis2(:,:, i))
    colormap('gray')
    axis off
end

H = uc_vectors(:, [1 2 4 6 9 20]);