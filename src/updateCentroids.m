function centroids = updateCentroids( net, imdb, featureLayerName, featureLayerSize, IDX, M )

mode = net.mode;
net.move('gpu');

trnIdx = imdb.images.set == 1;
nTrn = sum(trnIdx);
trnImg = imdb.images.data(:, :, :, trnIdx);
trnIdxViews = imdb.images.idxViews(trnIdx);
feature = zeros(featureLayerSize, nTrn, 'single');
net.mode = 'test';
layerIdx = net.getLayerIndex(featureLayerName);
idx = net.getVarIndex(net.layers(layerIdx).outputs);
net.vars(idx).precious = true;
for i = 1:100:nTrn
    upper = min(i+99, nTrn);
    image = trnImg(:, :, :, i:upper);
    idxViews = trnIdxViews(i:upper);
    image = gpuArray(image);
    net.eval({'image', image, 'idxViews', idxViews, 'M', M});
    feature(:, i:upper) = gather(net.vars(idx).value);    
end
K = 500;

centroids = zeros(featureLayerSize, K, 'single');
n = zeros(1, K);
% update centroids vectors
for idxTrn = 1:nTrn
    idxCluster = IDX(idxTrn);
    centroids(:, idxCluster) = centroids(:, idxCluster) + feature(:, idxTrn);
    n(idxCluster) = n(idxCluster) + 1;
end
for i = 1:K
    centroids(:, i) = centroids(:, i) / n(i);
end

centroids = gpuArray(single(centroids));
net.mode = mode;
end

