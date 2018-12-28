function [ CMC, mAP ] = evalCMCnMAP( dist, para, testingOption )

if strcmp(testingOption, 'Market') || strcmp(testingOption, 'MARS')
    IdGal = para.labelTest;
    nQuery = para.numTotalImgQuery;
    labelQuery = para.labelQuery;
    viewQuery = para.idxViewQuery;
    viewGal = para.idxViewTest;
    numTotalImgGal = para.numTotalImgTest;
elseif strcmp(testingOption, 'SYSUsingle')
    IdGal = 1:para.numTotalIdTest;
    nQuery = size(dist, 2);
    labelQuery = IdGal;
    viewQuery = 2*ones(1, para.numTotalIdQuery);
    viewGal = ones(1, para.numTotalIdTest);
    numTotalImgGal = size(dist, 1);
elseif strcmp(testingOption, 'SYSUmulti3')
    IdGal = repmat(1:para.numTotalIdTest, 3,1);
    IdGal = IdGal(:);
    nQuery = size(dist, 2);
    labelQuery = IdGal;
    viewQuery = 2*ones(1, 3*para.numTotalIdQuery);
    viewGal = ones(1, 3*para.numTotalIdQuery);
    numTotalImgGal = size(dist, 1);
elseif strcmp(testingOption, 'SYSUmultiAll')
    IdGal = para.labelTest(para.idxViewTest == 1);
    nQuery = size(dist, 2);
    labelQuery = para.labelQuery;
    viewQuery = para.idxViewQuery;
    viewGal = para.idxViewTest(para.idxViewTest == 1);
    numTotalImgGal = size(dist, 1);
end

junk0 = find(IdGal == -1);
ap = zeros(nQuery, 1);
CMC = zeros(numTotalImgGal, nQuery);

for i = 1:nQuery
    score = dist(:, i);
    q_label = labelQuery(i);
    q_cam = viewQuery(i);
    pos = find(IdGal == q_label);
    pos2 = viewGal(pos) ~= q_cam;
    good_image = pos(pos2);
    pos3 = viewGal(pos) == q_cam;
    junk = pos(pos3);
    junk_image = [junk0; junk];
    [~, index] = sort(score, 'ascend');
    [ap(i), CMC(:, i)] = compute_AP(good_image, junk_image, index);
end
CMC = sum(CMC, 2)./nQuery;
CMC = CMC';
mAP = sum(ap)/length(ap);

end