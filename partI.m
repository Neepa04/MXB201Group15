%%Variable definitions:
%%The signal S0 that was recorded for each voxel before any gradient pulses were applied.
%%The single b-value that was used for all gradient pulses (constant at 1000 s/mm).
%%The directions gi that were used for each gradient pulse.
%%For each gradient direction gi, the signal S that was recorded for each voxel.
% D is the diffusion tensor, a SPD 3x3 matrix.

%Create an overdetermined system, use least-squares to solve the system and retrieve the values for the diffusion tensor.

%dtmri contains arrays: (S,S0,b,g, mask)
%Mask is a binary array that differentiates the MRI area of interest (brain), and unneeded noise (background).


