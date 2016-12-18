function [nTrajectories] = extractIDTFeatures(dataPath, varargin)
% Extracts Improved Dense Trajectory features for every user with given
% sign range
%
% dataPath          : data set path
% indexRange        : samples indices (default = 1:2000)
% processType       : 'extract'     extract features from exe and given sample
%                     'decompress'  decompresses and reads extracted features
% saveFlag          : saves features after reading (default = false)
% verbose           : prints output to console

processType = [];
indexRange = [];
saveFlag = [];
verbose = [];
for i = 1:2:length(varargin),
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'processType'
            processType = value;
        case 'indexRange'
            indexRange = value;
        case 'saveFlag'
            saveFlag = value;
        case 'verbose'
            verbose = value;
        otherwise
    end
end

if isempty(processType), processType = 'extract'; end
if isempty(indexRange), indexRange = 1:2000; end
if isempty(saveFlag), saveFlag = false; end
if isempty(verbose), verbose = false; end
if nargin < 1, disp('dataPath argument cannot be empty'); return; end

executableName = 'DenseTrackStab.exe';
inputFileTemplateName = 'color.avi';
outputFile_compressed = 'color.features.gz';
outputFile_decompressed = 'color.features';

if strcmp(processType, 'read') == 1,
   features = []; 
end

userFolders = dir(fullfile(dataPath)); userFolders = userFolders(3:end);
for userIdx = 1:numel(userFolders),
    user = userFolders(userIdx);  
    
    if user.isdir == 1,
        nTrajectories = [];
        userDir = fullfile(dataPath, user.name);
        signFolders = dir(fullfile(userDir)); signFolders = signFolders(3:end);
        
        if exist([userDir filesep 'nTrajectories.mat'], 'file')
            load([userDir filesep 'nTrajectories.mat']);
        end
        
        for signIdx = indexRange,
            sign = signFolders(signIdx);
            
            currentSign = fullfile(userDir, sign.name);
            inputFile = dir(fullfile(currentSign, inputFileTemplateName));
            if isempty(inputFile) == 0,
               filePath = [userDir filesep sign.name];
               
               if verbose, fprintf('User: %d Sign: %d ', userIdx, signIdx); end
               tic;
               if strcmp(processType, 'extract') == 1,
                   fileInput = [filePath filesep inputFileTemplateName];
                   execStr = [executableName ' ' fileInput ' | gzip > ' filePath filesep outputFile_compressed];
                   system(execStr);
               else
                   execStr = [filePath filesep outputFile_compressed];
                   gunzip(execStr);             
                   execStr = [filePath filesep outputFile_decompressed];
                   feature = dlmread(execStr);
                                      
%                    load([filePath filesep num2str(userIdx) '-' num2str(signIdx) '.mat']);
                 
                   % deðiþecek
                   t = [userIdx, signIdx, size(feature(:, :), 1)];
                   nTrajectories = [nTrajectories; t];
                   
                   if saveFlag == true,
                      featureName = fullfile([filePath filesep num2str(userIdx) '-' num2str(signIdx) '.mat']);
                      save(featureName, 'feature', '-v7.3');
                   end
                   delete(execStr(1:end));
               end
               toc;
            end
        end
        
        if saveFlag == true,
            nTrajectoryName = fullfile([userDir filesep 'nTrajectories.mat']);
            save(nTrajectoryName, 'nTrajectories', '-v7.3');            
        end
    end    
end

end