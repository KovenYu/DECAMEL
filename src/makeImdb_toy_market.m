clear, clc
data = zeros(144, 56, 3, 0, 'single');
idxViews = zeros(1, 0, 'single');
labels = zeros(1, 0, 'single');
set =zeros(1, 0, 'uint8');

folder = {'train', 'test', 'query'};
for i = 1:numel(folder)
    path = ['../data/toy_market/', folder{i}];
    directory = dir(path);
    idx = arrayfun(@(x)length(x.name)<5||~strcmp(x.name(5), '_'), directory);
    directory = directory(~idx);
    n = numel(directory);
    data_t = zeros(144,56,3,n,'single');
    labels_t = zeros(1, n, 'single');
    idxViews_t = zeros(1, n, 'single');
    set_t = ones(1, n, 'uint8');
    for j = 1:n
        filename = directory(j).name;
        labels_t(j) = single(str2double(filename(1:4)));
        idxViews_t(j) = single(str2double(filename(7)));
        filepath = [path, '/', filename];
        data_t(:, :, :, j) = jstl_imread(filepath);
        set_t(j) = i;
    end
    if i > 1 % test(gal, 3) or query(prb, 4)
        set_t = set_t + 1;
    end
    data = cat(4, data, data_t);
    labels = cat(2, labels, labels_t);
    set = cat(2, set, set_t);
    idxViews = cat(2, idxViews, idxViews_t);
end

images.data = data;
images.labels = labels;
images.set = set;
images.idxViews = idxViews;
save('../data/toy_market.mat', 'images', '-v7.3')