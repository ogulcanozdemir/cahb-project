function [trajectoryMap, numberOfTrajectories] = mapTrajectories(dataPath, dataRange, varargin)
% Maps trajectories of extracted features according to their 
% user, sign and number of trajectories
% dataPath      : data set path
% dataRange     : data range for mapping index

saveName = [];
saveFlag = [];
for i = 1:2:length(varargin),
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'save'
            saveName = value; 
            saveFlag = true;
        otherwise
    end
end

if isempty(saveName), saveFlag = false; end
if nargin < 2, disp('dataRange cannot be empty'); return; end
if nargin < 1, disp('dataPath and dataRange cannot be empty'); return; end

disp('Started mapping trajectories...');
trajectoryFileName = 'nTrajectories.mat';

userFolders = dir(dataPath); 
userFolders = userFolders(3:end);
userFolders = userFolders(dataRange);

tic;
trajectoryMap = [];
numberOfTrajectories = [];
for userIdx = 1:numel(userFolders),
   user = userFolders(userIdx);
   load([dataPath filesep user.name filesep trajectoryFileName]); % loads nTrajectories matrix
   
   numberOfTrajectories = [numberOfTrajectories; nTrajectories(:, :)]; 
   for signIdx = 1:size(nTrajectories, 1),
      t = repmat(nTrajectories(signIdx, 1:2), nTrajectories(signIdx, 3), 1); 
      t(:, 3) = 1:nTrajectories(signIdx, 3);
      trajectoryMap = [trajectoryMap; t];
   end
end
numberOfTrajectories(:, 3) = cumsum(numberOfTrajectories(:, 3));
toc;

if saveFlag == true,
    save([pwd filesep 'temp' filesep 'map' filesep ... 
        'trajectoryMap_' saveName '.mat'], 'trajectoryMap', 'numberOfTrajectories', '-v7.3');
end

end