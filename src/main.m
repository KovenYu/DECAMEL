% if you find our work helpful in your research,
% please kindly cite our paper:
% @article{yu2018unsupervised,
% title={Unsupervised Person Re-identification by Deep Asymmetric Metric Embedding},
% author={Yu, Hong-Xing and Wu, Ancong and Zheng, Wei-Shi},
% journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (DOI 10.1109/TPAMI.2018.2886878)},
% year={2019},
% }

function main(varargin)

lambda = 1e-2;
gamma = 10;

run('../matconvnet-1.0-beta23/matlab/vl_setupnn.m')
opts.expDir = '../exp';
opts.gpus = [];
opts.prepareGPU = true;
opts.data_path = 'market.mat';
opts = vl_argparse(opts, varargin); 

if opts.prepareGPU
    prepareGPUs(opts, true);
end

%% load IMDB
imdbPath = ['../data/', opts.data_path];
imdb = load(imdbPath);
nViews = numel(unique(imdb.images.idxViews));

%% load original model
baseModelPath = '../data/pretrained_resnet56.mat';
baseModel = load(baseModelPath);
net = dagnn.DagNN.loadobj(baseModel.net);
featureLayerName = 'pool_final';
idx = net.getLayerIndex(featureLayerName);
net.layers(idx+1:end) = [];
net.rebuild();
featureLayerSize = 64;
opts.feaSize = featureLayerSize;

%% pre-run CAMEL
[Us, IDX, M] = CAMEL(net, imdb, featureLayerName, featureLayerSize, lambda);
opts.IDX = IDX;
opts.M = gpuArray(single(M));

%% add new layers
block = dagnn.Mork('size', [1, 1, featureLayerSize, featureLayerSize], ...
    'stride', 1, 'pad', 0, 'nViews', nViews, 'lambda', lambda, 'gamma', gamma);
U = {};
for i = 1:nViews
    U{i} = sprintf('U%d', i);
end
net.addLayer('U', block, {featureLayerName, 'idxViews', 'M'}, 'U', U);
block.setParams(Us);
featureLayerName = 'U';
opts.featureLayerName = featureLayerName;
opts.centroids = updateCentroids(net, imdb, featureLayerName, featureLayerSize, IDX, M);
net.addLayer('ClusteringLoss', dagnn.ClusteringLoss(),...
    {'U', 'centroids', 'IDX'}, 'ClusteringLoss');

%% prepare training params and data

net.meta.trainOpts.learningRate = 5e-3*[ones(1,100), 0.2*ones(1,100)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate);
net.meta.trainOpts.weightDecay = 0;
net.meta.trainOpts.batchSize = 216;
% net.conserveMemory = false;

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
  net.meta.trainOpts, ...
  opts, ...
  'val', find(imdb.images.set == 3), ...
  'derOutputs', {'ClusteringLoss', 1});

[rank1, rank5, rank10, map] = evalRank1(net, opts, imdb);
fprintf('rank1: %.2f, rank5: %.2f, rank10: %.2f, map: %.2f\n', rank1, rank5, rank10, map);

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% % -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch); 
idxViews = imdb.images.idxViews(batch);
if opts.numGpus >= 1
    images = gpuArray(images);
end
inputs = {'image', images, 'idxViews', idxViews} ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;
