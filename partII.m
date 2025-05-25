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
title('Mean Face')

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

randomnum = randi(1000);
random_face = A(:, randomnum);         % Random face

S = U(:, 1:20);                        % Eigenface space of largest singular values
Proj_S = S*inv(S'*S)*S';               % Projector onto the eigenface space


% Coordinate vector (Least Squares Solution)
LHS = S'*S;
RHS = S'*random_face;
c_vector = LHS \ RHS;

recon_face = S*c_vector;               % Reconstructed face using linear combinations of eigenfaces

% Visualisation of original face and reconstructed face
figure      % Original Random Face
randomface = reshape(uint8(random_face), rows, cols);
imshow(randomface, 'InitialMagnification', 'Fit')
title('Original Face')

figure      % Reconstructed Face
face_recon = reshape(uint8(recon_face), rows, cols);
imshow(face_recon, 'InitialMagnification', 'Fit')
title('Reconstructed Face')

%% Demonstrate rudimentary moustache detector

% The 13th column of the eigenface space corresponds to the moustache characteristic
% Therefore, the 13th row of the coordinate vector determines the level of moustache apparent in the photo

if c_vector(13) > 0.5
    disp("Moustache detected")
else
    disp("Moustache Undetected")
end