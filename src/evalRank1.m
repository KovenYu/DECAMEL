% -------------------------------------------------------------------------
function [rank1, rank5, rank10, map] = evalRank1(net, params, imdb)
% -------------------------------------------------------------------------

feaSize = params.feaSize;
featureLayerName = params.featureLayerName;

gal = imdb.images.set == 3;
prb = imdb.images.set == 4;

imgPrb = imdb.images.data(:, :, :, prb);
idxPrb = imdb.images.idxViews(prb);
imgGal = imdb.images.data(:, :, :, gal);
idxGal = imdb.images.idxViews(gal);
labelPrb = imdb.images.labels(1, prb);
labelGal = imdb.images.labels(1, gal);

% layerIdx = net.getLayerIndex('ClusteringLoss');
% layerIdx = layerIdx - 1;
layerIdx = net.getLayerIndex(featureLayerName);
idx = net.getVarIndex(net.layers(layerIdx).outputs);
net.vars(idx).precious = true;

nPrb = size(imgPrb, 4);
nGal = size(imgGal, 4);

feaPrb = zeros(feaSize, nPrb);
feaGal = zeros(feaSize, nGal);
for i = 1:100:nPrb
    net.mode = 'test';
    upper = min(i+99, nPrb);
    image = imgPrb(:, :, :, i:upper);
%     label = labelPrb(1, i:upper);
    idxViews = idxPrb(1, i:upper);
    if strcmp(net.device, 'gpu')
        image = gpuArray(image);
    end
    net.eval({'image', image, 'idxViews', idxViews, 'M', params.M, ...
        'IDX', 1:nPrb, 'centroids', params.centroids});
    feaPrb(:, i:upper) = gather(net.vars(idx).value);
end
for i = 1:100:nGal
    net.mode = 'test';
    upper = min(i+99, nGal);
    image = imgGal(:, :, :, i:upper);
%     label = labelGal(1, i:upper);
    idxViews = idxGal(1, i:upper);
    if strcmp(net.device, 'gpu')
        image = gpuArray(image);
    end
    net.eval({'image', image, 'idxViews', idxViews, 'M', params.M, ...
        'IDX', 1:nGal, 'centroids', params.centroids});
    feaGal(:, i:upper) = gather(net.vars(idx).value);
end
dist = pdist2(feaGal', feaPrb', 'cosine');
% [~, rankingTable] = sort(dist);
para.labelTest = labelGal;
para.labelQuery = labelPrb;
para.numTotalImgQuery = nPrb;
para.idxViewQuery = idxPrb;
para.idxViewTest = idxGal;
para.numTotalImgTest = nGal;

[CMC, map] = evalCMCnMAP(dist, para, 'Market');
rank1 = CMC(1)*100;
rank5 = CMC(5)*100;
rank10 = CMC(10)*100;
map = map*100;

net.vars(idx).precious = false;
