clc;
clear;
close all;

% assignment 06

Igray = imread('ann/training.jpg');

BW = ~im2bw(Igray); 

SE = strel('disk',1);
BW2 = imerode(BW, SE); 

labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 1000 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);

for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW2(row1:row2, col1:col2);
    
    subImageScaled = imresize(subImage, [24 12]);

    TPattern(k,:) = subImageScaled(:)';
end

TTarget = zeros(100,10);

for row = 1:10
    for col = 1:10
        TTarget(10*(row - 1) + col,row) = 1;
    end
end

TPattern = TPattern';
TTarget = TTarget';

net = newff([zeros(288,1), ones(288,1)], [24,10], {'logsig', 'logsig'}, 'traingdx');
net.trainParam.epochs = 500;
net = train(net, TPattern, TTarget);

%% Pattern

Igray = imread('ann/196128.jpg');

BW = ~im2bw(Igray); 

SE = strel('disk',1);
BW2 = imerode(BW, SE); 

labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 1000 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);

for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW2(row1:row2, col1:col2);
    
    subImageScaled = imresize(subImage, [24 12]);

    UPattern(k,:) = subImageScaled(:)';
    
end

UPattern = UPattern';

Y = sim(net, UPattern);

