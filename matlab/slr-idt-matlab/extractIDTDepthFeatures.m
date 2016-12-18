clear all, close all, clc

fileDirectory = 'C:\Users\ogulcanozdemir\Desktop\';
load([fileDirectory filesep 'depth.mat']);

figure, set(gcf, 'Color', 'white'); % axis tight
set(gca, 'Visible','off');
depthVideo = VideoWriter([fileDirectory filesep 'depthVideo_1.avi']);
depthVideo.Quality = 100;
open(depthVideo);
for f=1:size(depthMat,3),
    imagesc(depthMat(:, :, f));
    frame = getframe(gca);
    
    img = frame.cdata;
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    
    medfilimg(:, :, 1) = medfilt2(R, [5 5]);
    medfilimg(:, :, 2) = medfilt2(G, [5 5]);
    medfilimg(:, :, 3) = medfilt2(B, [5 5]);
    
    writeVideo(depthVideo, medfilimg);
end
close(gcf), close(depthVideo);
