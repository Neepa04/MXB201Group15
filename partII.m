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
mean_face = sum(A,2) ./ size(A, 2);

% Visualising mean face
mean_face_vis = reshape(uint8(mean_face), rows, cols);
imshow(mean_face_vis, 'InitialMagnification', 'Fit')

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

for x = 1:20
    nexttile
    imagesc(eigenfaces(:,:,x));
    colormap('gray')
    title(num2str(x))
    axis off
end

%% Calculate coordinate vectors

random_face = A(:, 378);     % Random face

S = U(:, 1:47);              % Eigenface space of largest singular values

Proj_S = S*inv(S'*S)*S';       % Projector onto the eigenface space


% Coordinate vector
Proj_S_face = Proj_S*random_face;

Proj_S_face_vis = reshape(uint8(Proj_S_face), rows, cols);

figure
imshow(Proj_S_face_vis, 'InitialMagnification', 'Fit')

figure
randomface = reshape(A(:,378), rows, cols);
imshow(uint8(randomface), 'InitialMagnification', 'Fit')





%% Demonstrate rudimentary moustache detector
