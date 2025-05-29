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
        if S0(x,y) <=1 %Skips to next loop if S0 is negative.
            continue;
        end
        
        if any(S(x,y,:)<=1, 'all') % Ends loop if any values are negative for a given x,y
            continue;
        end

        % Solving least squares problem
        B = -log (squeeze(S(x,y,:))./ S0(x,y)) / b; %Find B

        D_vector = A\B; %Use MATLAB \ function to find least squares D vector
       
        % Forming diffusion tensor
        D = [D_vector(1) D_vector(4) D_vector(5); D_vector(4), D_vector(2), D_vector(6); D_vector(5), D_vector(6), D_vector(3)]; %Arrange D values into 3x3 matrix as per guidelines
        
        % Finding eigenvalues and eigenvectors
        [EVec,EVal] = eig(D);
        lambda = diag(EVal); % Retrieve diagonal of EVal which are D's eigenvalues
      
        % Calculating MD, FA and PDD
        % MD calculation
        if mean(lambda) > 10^-9 % Ensure very small numbers are not written.
        MD(x,y) = mean(lambda); % Enscribe average of eigenvalues before restarting loop.
        end
        % FA calculation
        FA(x,y) = (sqrt(3)./sqrt(2)) .* sqrt((lambda(1,1)-MD(x,y)).^2 + (lambda(2,1)-MD(x,y)).^2 + (lambda(3,1)-MD(x,y)).^2)./(sqrt(lambda(1,1).^2+lambda(2,1).^2+lambda(3,1).^2)); % Apply formula form Jiang et al.

        % PDD calculation
        [~,big_l] = max(lambda); % Return index of largest eigenvalue
        big_v = EVec(:,big_l); % Return eigenvector of largest eigenvalue
        PDD(x,y,:) = big_v ./ norm(big_v);  % Normalise and store eigenvector of largest eigenvalue
    end
end

%% Plot mean diffusivity, fractional anisotropy and principal diffusion direction maps
figure('Name','DTI maps');

subplot(1,3,1);
imagesc(MD); axis image off; colormap(gray);
title('Mean Diffusivity');

subplot(1,3,2);
imagesc(FA,[0 1]); axis image off; colormap(gray);
title('Fractional Anisotropy');

subplot(1,3,3);
rgb = abs(PDD);          
rgb(:,:,1) = rgb(:,:,1).*FA;
rgb(:,:,2) = rgb(:,:,2).*FA;
rgb(:,:,3) = rgb(:,:,3).*FA;
imagesc(rgb); axis image off;
title('Principal Diffusion Direction');
