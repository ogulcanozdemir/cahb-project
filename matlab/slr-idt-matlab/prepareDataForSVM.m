function prepareDataForSVM(fisherDataPath, repeatCount, outputPath, outputFileName)
% prepares fisher data for classification
% fisherDataPath        : extracted fisher vectors path
% repeatCount           : repeat count of processes
% outputPath            : path of result data
% outputFileName        : name of the output file

if nargin < 4, outputFileName = 'svm_data_1000_500s'; end
if nargin < 3, outputPath = pwd; end
if nargin < 2, repeatCount = 5; end
if nargin < 1, fisherDataPath = pwd; end

load(fisherDataPath);
for repeatIdx=1:repeatCount,
    tmpData = modeledData{1, repeatIdx};
    
    data = [];
    labels = [];
    for userIdx = 1:size(tmpData, 1),
        for signIdx = 1:size(tmpData, 2),
            labels = [labels; signIdx];
            data = [data; tmpData{userIdx, signIdx}'];
        end
    end
    
    clear test_data;
    save([outputPath filesep outputFileName '_' num2str(repeatIdx)], 'data', 'labels', '-v7.3');
end

end