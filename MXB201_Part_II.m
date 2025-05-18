% Template for MXB201 Project Part II.

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
mean_face = sum(A, 2) ./ size(A, 2);    % Finding the average column of A

% Converting the dimensions to a readbile matrix for visualisation
mean_face_vis = zeros(192, 168);

for j = 1:168
    mean_face_vis(:,j) = mean_face( ((j - 1) * 192) + 1:j * 192);
end

imshow(uint8(mean_face_vis), 'InitialMagnification','fit')

%% Calculate mean-centred SVD

%% Visualise first 20 eigenfaces

%% Calculate coordinate vectors

%% Demonstrate rudimentary moustache detector
