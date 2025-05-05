%%Variable definitions:
%%The signal S0 that was recorded for each voxel before any gradient pulses were applied.
%%The single b-value that was used for all gradient pulses (constant at 1000 s/mm).
%%The directions gi that were used for each gradient pulse.
%%For each gradient direction gi, the signal S that was recorded for each voxel.
% D is the diffusion tensor, a SPD 3x3 matrix.

%Create an overdetermined system, use least-squares to solve the system and retrieve the values for the diffusion tensor.

%dtmri contains arrays: (S,S0,b,g, mask)
%Mask is a binary array that differentiates the MRI area of interest (brain), and unneeded noise (background).


% Template for MXB201 Project Part I.

%% Initialisation
clear
load partI.mat

[X,Y,num_dirs] = size(S);
assert(isequal(size(g), [num_dirs 3]));

% These arrays will be be filled in during the main loop below
MD  = nan(X, Y);    % mean diffusion
FA  = nan(X, Y);    % fractional anistropy
PDD = nan(X, Y, 3); % principal diffusion direction

%Any other initialisation needed here
A =[g(:,1).^2,g(:,2).^2 , g(:,3).^2 , 2*g(:,1).*g(:,2),2*g(:,1).*g(:,3),2*g(:,2).*g(:,3)]; %initialises A matrix for rearranged eq.

%% Compute the diffusion tensor for each pixel
for x = 1:X
    for y = 1:Y

        % If not within the mask, skip the pixel
        if ~mask(x, y), continue; end
        
        % Handling bad data 
            %S and S0 measurements should not be negative (measurements cannot be negative and logs will be not computable.)
        if S0(x,y) <= 0 %Skips to next loop if S0 is negative.
            continue;
        end
        
        if any(S(x,y,:)<=0, 'all') %Ends loop if any values are negative for a given x,y
            continue
        end

        % Solving least squares problem
        B = -log (squeeze(S(x,y,:))./ S0(x,y)) / b; %Find B

        D_vector = A\B; %Use MATLAB \ function to find least squares D vector
       
        % Forming diffusion tensor
        D = [D_vector(1) D_vector(4) D_vector(5); D_vector(4), D_vector(2), D_vector(6); D_vector(5), D_vector(6), D_vector(3)]; %Arrange D values into 3x3 matrix as per guidelines
        
        % Finding eigenvalues and eigenvectors
        [EVec,EVal] = eig(D)
        lambda = diag(EVal); %Retrieve diagonal of EVal which are D's eigenvalues
      
        % Calculating MD, FA and PDD
        MD(x,y) = mean(lambda); % Enscribe average of eigenvalues before restarting loop.

    end
end

%% Plot mean diffusivity, fractional anisotropy and principal diffusion direction maps

Trimmed_MD = prctile(MD(mask),99);
imagesc(MD,[0 Trimmed_MD])
axis image off
colormap(gray), colorbar





