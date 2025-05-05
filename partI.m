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
        Measurement_count = size(S,3);
        for i = 1:Measurement_count
            if S(x,y,i) <= 0
                continue;
            end
        end

        % Solving least squares problem
        B = log (squeeze(S(x,y,:))./ S0(x,y)) / b;

        D = A\B
        % Forming diffusion tensor
        % Finding eigenvalues and eigenvectors
        % Calculating MD, FA and PDD

    end
end

%% Plot mean diffusivity, fractional anisotropy and principal diffusion direction maps
