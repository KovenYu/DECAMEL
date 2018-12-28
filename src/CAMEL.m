function [ Us, IDX, M ] = CAMEL( net, imdb, featureLayerName, featureLayerSize, lambda )
% output:
%  Us: cell array, containing {U1, U2}. U1 double, d*T
%  IDX: clustering results
%  M: double, covariance matrix
rng(1);
mode = net.mode;

%% extract feature
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
%     idxViews = trnIdxViews(i:upper);
    image = gpuArray(image);
    net.eval({'image', image});
    feature(:, i:upper) = gather(net.vars(idx).value);    
end

%% perform CAMEL
k = 500;
alpha = 1;
nViews = numel(unique(imdb.images.idxViews));
nBasis = featureLayerSize;
maxIter = 100;
options = statset('MaxIter', 20);
data = feature;
dataAsym = AsymShift(data, trnIdxViews, nViews);

IDX = kmeans(data', k, 'options', options);
L = IDX2L(IDX);
M = constructM(data, trnIdxViews, alpha, featureLayerSize, nViews);
I = constructI(featureLayerSize, nViews);

n = nTrn;
fclustering = (trace(data*data') - trace(L'*(data'*data)*L)) / n;
fprintf('fclustering = %f \n', fclustering);
temp = (dataAsym*dataAsym' - dataAsym*(L*L')*dataAsym') / n;
MATRIX = M \ (lambda*I + temp);
[V,D] = eig(MATRIX);
U = constructU(V, D, nBasis, M, nViews);

fU = lambda*trace(U'*I*U);
fclustering = (trace((U*U')*(dataAsym*dataAsym')) - trace(L'*dataAsym'*(U*U')*dataAsym*L)) / n;
fobj = fU + fclustering;
fprintf('fobj = %f, fU = %f, fclustering = %f\n', fobj, fU, fclustering);

%% iterative optimization
fobjPrevious = fobj;
for iter = 1:maxIter
    XprojT = (U'*dataAsym)';
    
    IDX = kmeans(XprojT, k, 'options', options);
    
    L = IDX2L(IDX);
    if any(any(isnan(L))) % a matlab bug.
        IDX = kmeans(XprojT, k , 'options', options);
        L = DIX2L(IDX);
    end
    
    fU = lambda*trace(U'*I*U);
    fclustering = (trace(dataAsym'*(U*U')*dataAsym) - trace(L'*dataAsym'*(U*U')*dataAsym*L)) / n;
    fobj = fU + fclustering;
    fprintf('fobj = %f, fU = %f, fclustering = %f\n', fobj, fU, fclustering);
    
    MATRIX = M \ (lambda*I +(dataAsym*dataAsym' - dataAsym*(L*L')*dataAsym') / n);
    
    [V,D] = eig(MATRIX);
    U = constructU(V, D, nBasis, M, nViews);
    
    fU = lambda*trace(U'*I*U);
    fclustering = (trace(dataAsym'*(U*U')*dataAsym) - trace(L'*dataAsym'*(U*U')*dataAsym*L)) / n;
    fobj = fU + fclustering;
    fprintf('fobj = %f, fU = %f, fclustering = %f\n', fobj, fU, fclustering);
    
    if (fobj - fobjPrevious) > 0
        break
    end
    fobjPrevious = fobj;
    if iter == maxIter
        warning('The default maximum # of iterations has been reached.');
    end

end
Us = {};
d = featureLayerSize;
T = size(U, 2);
for i = 1:nViews
    Ui = U(1+(i-1)*d : i*d, :);
    Ui = single(reshape(Ui, [1, 1, d, T]));
    Us = [Us, Ui];
end
net.mode = mode;

function shiftedData = AsymShift( originalData, idxView, numViews )
% input :   originalData,  d by n
% output:   shiftedData,  para.numViews*d by n
% shift training data to fit the combined projection matrix U

[d, n] = size(originalData);
shiftedData = zeros(numViews*d, n);
for i = 1:numViews;
    idx = (idxView == i);
    l = sum(idx); % number of the imgs under the i-th views
    if i == 1
        data = [originalData(:,idx); zeros( (numViews-1)*d, l)];
    elseif i == numViews
        data = [zeros( (numViews-1)*d, l); originalData(:,idx)];
    else
        data = [zeros( (i-1)*d, l); originalData(:,idx); zeros( (numViews-i)*d, l)];
    end
    shiftedData(:,idx) = data;
end

function L = IDX2L( IDX )

% input   IDX, 1*n, every element denotes which centroid this element
% belongs to
% output  L, n*k

k = unique(IDX);
k = k(end);
n = length(IDX);
L = zeros(n,k);
for i = 1:k
    temp = IDX == i;
    L(:,i) = temp/ sqrt(sum(temp));
end

function M = constructM( data, idxView, alpha, d, numViews )
% input :  data,  d by n
% output:  M = [M1 0  ... 0
%               0  M2 ... 0
%               0  0  ... Mv]
%        where Mi = cov of data in i-th view.  d by d

M = zeros(d*numViews);

for i = 1:numViews
    idx = (idxView == i);
    tempX = data(:, idx);
    ni = size(tempX, 2);
    M(1 + (i-1)*d : i*d, 1 + (i-1)*d : i*d) = tempX*tempX' / ni; % Mi
end

M = M + alpha*trace(M)/size(M,1)*eye(size(M));

function I = constructI( d, N )
% I = [(N-1)E -E -E ... -E   \
%      -E (N-1)E -E ... -E   |
%               .            |
%               .            |-> N
%               .            |
%      -E -E ...    (N-1)E]; /

E = eye(d);
minusBigE = repmat(-E, N, N);
I = minusBigE + N*eye(N*d);

function U = constructU( V, D, nBasis, M, N )
% s.t. U'MU = NE
%   => u'Mu = N

% find the smallest eigens
d = diag(D);
[~, idx] = sort(d);
U_unnormalized = V(:, idx(1:nBasis));

U = zeros(size(U_unnormalized));
n = size(U, 2);
for i = 1:n
    u_un = U_unnormalized(:, i);
    u = u_un*sqrt(N)/sqrt(u_un'*M*u_un);
    U(:,i) = u;
end
U = real(U);