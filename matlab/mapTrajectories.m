function [trajectoryMap, numberOfTrajectories] = mapTrajectories(featurePath, dataRange, nTrajectories)

numberOfTrajectories = [numberOfTrajectories; nTrajectories(:, :)]; 
for signIdx = 1:size(nTrajectories, 1),
  t = repmat(nTrajectories(signIdx, 1:2), nTrajectories(signIdx, 3), 1); 
  t(:, 3) = 1:nTrajectories(signIdx, 3);
  trajectoryMap = [trajectoryMap; t];
end

end