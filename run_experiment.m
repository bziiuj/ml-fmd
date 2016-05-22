%--------------------------------------------------
function run_experiment()
%--------------------------------------------------
  setup;

  opts.visualize = false;
  opts.slicesNumber = 10;
  opts.numberOfPeaks = 25; % Hough parameter
  opts.threshold = 0.3; % Hough parameter
  opts.minLength = 0.2; % Hough parameter
  opts.trainSetSize = 40;
  opts.trainFramesPerVideo = 100;
  opts.intersectionRatio = 0.5; % how much gtLine and houghLine have to overlap to be matched
  opts.maxDifference = 3; % maximal mean difference between matched gtLine and houghLine
  opts.equalPosNegCount = false; % take as much negatives as positives
  opts.lineDescriptor = 'TinyImage'; %'ProfilePlot', 'BriefLike'
  opts.minLeafSize = 5; % RUSBoost
  opts.numberOfTrees = 500; % RUSBoost
  opts.learnRate = 0.1; % RUSBoost
  opts.profilePlotLength = 100;
  opts.tinyImageWidth = 32;
  opts.tinyImageHeight = 4; % height in output code
  opts.tinyImageLineHeight = 10; % height in original image
  opts.briefLikeWidth = 100;
  opts.briefLikeHeight = 10;
  opts.briefLikePairs = 128;
  opts.briefLikeSigma = 0.5;
  opts.testLearners = 100; % based on the results from generate classifier state

  dirs.anonymousDir = 'data/anonymous';
  dirs.expDir = fullfile(dirs.anonymousDir, ['exp', opts.lineDescriptor]);
  dirs.roiDir = fullfile(dirs.anonymousDir, 'roi');
  dirs.posNegLinesDir = fullfile(dirs.anonymousDir, 'posNegLines');
  dirs.imdbDir = fullfile(dirs.anonymousDir, 'imdb');

  if ~exist(dirs.expDir)
    mkdir(dirs.expDir);
  end
  if ~exist(dirs.posNegLinesDir)
    mkdir(dirs.posNegLinesDir);
  end
  if ~exist(dirs.imdbDir)
    mkdir(dirs.imdbDir);
  end

  %poolobj = parpool('local', opts.slicesNumber) ;
  %parfor (seed = 1:opts.slicesNumber, opts.slicesNumber)
  for seed = 1:opts.slicesNumber
    fprintf('%d: Begin of seed\n', seed)
    runExperimentSeed(seed, dirs, opts);
    fprintf('%d: End of seed\n', seed)
  end
  %delete(poolobj) ;
end


function runExperimentSeed(seed, dirs, opts)
  opts.seed = seed;

  imdbFilePath = fullfile(dirs.imdbDir, ['imdb-', num2str(opts.seed), '.mat']);
  expSeedDir = fullfile(dirs.expDir, ['seed-', num2str(opts.seed)]);

  % Generating (or loading) training set (random videos and random
  % frames)
  if ~exist(imdbFilePath)
    rng(opts.seed);
    trainSeed.trainSetIndices = sort(randsample(1:70, opts.trainSetSize));
    trainSeed.testSetIndices = setdiff(1:70, trainSeed.trainSetIndices);

    rng(opts.seed);
    for t = 1:length(trainSeed.trainSetIndices)
      trainIndex = trainSeed.trainSetIndices(t);
      load(fullfile(dirs.roiDir, ['rois-', num2str(trainIndex), '.mat']));

      trainSeed.examinations(t).trainIndex = trainIndex;
      trainSeed.examinations(t).frameIndices = sort(randsample(cat(1, roiInfos.frameIndex), opts.trainFramesPerVideo));
    end

    save(imdbFilePath, 'trainSeed');
    fprintf('%d: Saving %s\n', opts.seed, imdbFilePath)
  else
    load(imdbFilePath);
    fprintf('%d: Loading %s\n', opts.seed, imdbFilePath)
  end

  % Generating (or loading) lines with values
  posNetLineInfosFilePath = fullfile(dirs.posNegLinesDir, ['posNegLines-', num2str(opts.seed), '.mat']);
  if ~exist(posNetLineInfosFilePath)
    posNegLineInfos = repmat(struct('trainIndex', {}, 'frameIndex', {}, 'fInRoiInfos', {}, ...
      'posLowerLine', {}, 'posUpperLine', {}, 'negLowerLine', {}, 'negUpperLine', {}), ...
      1, opts.trainSetSize*opts.trainFramesPerVideo);
    for t = 1:opts.trainSetSize
      trainInfo = trainSeed.examinations(t);
      load(fullfile(dirs.roiDir, ['rois-', num2str(trainInfo.trainIndex), '.mat']));

      for f = 1:opts.trainFramesPerVideo
        fInRoiInfos = find(cat(1, roiInfos.frameIndex) == trainInfo.frameIndices(f));
        roiInfo = roiInfos(fInRoiInfos);

        roi = roiInfo.roi;
        gtLowerLine = roiInfo.gtLowerLine;
        gtUpperLine = roiInfo.gtUpperLine;

        houghLines = getHoughLines(roi, opts);

        [posLowerLine, posUpperLine, negLowerLine, negUpperLine] = ...
          matchLines(roi, gtLowerLine, gtUpperLine, houghLines, opts);

        index = (t-1)*opts.trainFramesPerVideo+f;
        posNegLineInfos(index).trainIndex = trainInfo.trainIndex;
        posNegLineInfos(index).frameIndex = trainInfo.frameIndices(f);
        posNegLineInfos(index).fInRoiInfos = fInRoiInfos;
        posNegLineInfos(index).posLowerLine = posLowerLine;
        posNegLineInfos(index).posUpperLine = posUpperLine;
        posNegLineInfos(index).negLowerLine = negLowerLine;
        posNegLineInfos(index).negUpperLine = negUpperLine;
      end
    end

    save(posNetLineInfosFilePath, 'posNegLineInfos');
    fprintf('%d: Saving %s\n', opts.seed, posNetLineInfosFilePath)
  else
    load(posNetLineInfosFilePath);
    fprintf('%d: Loading %s\n', opts.seed, posNetLineInfosFilePath)
  end

  % Generating (or loading) features
  posNetLineCodesFilePath = fullfile(expSeedDir, 'posNegLineCodes.mat');
  if ~exist(posNetLineCodesFilePath)
    if ~exist(expSeedDir)
      mkdir(expSeedDir);
    end
    
    % get descriptor
    desc = struct();
    switch opts.lineDescriptor
      case 'BriefLike'
        desc = getBriefLikeDescriptor(opts);
    end
    opts.desc = desc;
    
    % compute number of lower and upper lines
    lowerLineCodesLength = 0;
    upperLineCodesLength = 0;
    for trainIndex = cat(1, trainSeed.examinations.trainIndex)'
      lInPosNegLines = find(cat(1, posNegLineInfos.trainIndex) == trainIndex);
      for lInPosNegLine = lInPosNegLines'
        posLowerLines = posNegLineInfos(lInPosNegLine).posLowerLine;
        negLowerLines = posNegLineInfos(lInPosNegLine).negLowerLine;
        posUpperLines = posNegLineInfos(lInPosNegLine).posUpperLine;
        negUpperLines = posNegLineInfos(lInPosNegLine).negUpperLine;

        lowerLineCodesLength = lowerLineCodesLength + length(posLowerLines);
        lowerLineCodesLength = lowerLineCodesLength + length(negLowerLines);
        upperLineCodesLength = upperLineCodesLength + length(posUpperLines);
        upperLineCodesLength = upperLineCodesLength + length(negUpperLines);
      end
    end

    lowerLineCodesCell = cell(lowerLineCodesLength, 2);
    upperLineCodesCell = cell(upperLineCodesLength, 2);
    lowerLineIndex = 1;
    upperLineIndex = 1;
    for trainIndex = cat(1, trainSeed.examinations.trainIndex)'
      load(fullfile(dirs.roiDir, ['rois-', num2str(trainIndex), '.mat']));

      lInPosNegLines = find(cat(1, posNegLineInfos.trainIndex) == trainIndex)';
      for lInPosNegLine = lInPosNegLines
        posNegLineInfo = posNegLineInfos(lInPosNegLine);
        posLowerLines = posNegLineInfo.posLowerLine;
        negLowerLines = posNegLineInfo.negLowerLine;
        posUpperLines = posNegLineInfo.posUpperLine;
        negUpperLines = posNegLineInfo.negUpperLine;

        roi = roiInfos(posNegLineInfo.fInRoiInfos).roi;

        for posLowerLine = posLowerLines
          lowerLineCodesCell{lowerLineIndex, 1} = getLineDescriptor(roi, posLowerLine, opts);
          lowerLineCodesCell{lowerLineIndex, 2} = 1;
          lowerLineIndex = lowerLineIndex + 1;
        end
        for negLowerLine = negLowerLines
          lowerLineCodesCell{lowerLineIndex, 1} = getLineDescriptor(roi, negLowerLine, opts);
          lowerLineCodesCell{lowerLineIndex, 2} = 0;
          lowerLineIndex = lowerLineIndex + 1;
        end
        for posUpperLine = posUpperLines
          upperLineCodesCell{upperLineIndex, 1} = getLineDescriptor(roi, posUpperLine, opts);
          upperLineCodesCell{upperLineIndex, 2} = 1;
          upperLineIndex = upperLineIndex + 1;
        end
        for negUpperLine = negUpperLines
          upperLineCodesCell{upperLineIndex, 1} = getLineDescriptor(roi, negUpperLine, opts);
          upperLineCodesCell{upperLineIndex, 2} = 0;
          upperLineIndex = upperLineIndex + 1;
        end
      end
    end

    lowerLineCodes = cell2mat(lowerLineCodesCell(:, 1));
    lowerLineClass = cell2mat(lowerLineCodesCell(:, 2));
    upperLineCodes = cell2mat(upperLineCodesCell(:, 1));
    upperLineClass = cell2mat(upperLineCodesCell(:, 2));

    save(posNetLineCodesFilePath, 'lowerLineCodes', 'lowerLineClass', ...
      'upperLineCodes', 'upperLineClass', 'desc');
    fprintf('%d: Saving %s\n', opts.seed, posNetLineCodesFilePath)
  else
    load(posNetLineCodesFilePath);
    if exist('desc')
      opts.desc = desc;
    end
    fprintf('%d: Loading %s\n', opts.seed, posNetLineCodesFilePath)
  end

  % Generating (or loading) classifier
  classifierFilePath = fullfile(expSeedDir, 'classifier.mat');
  if ~exist(classifierFilePath)
    lowerClassifier = getClassifier(lowerLineCodes, lowerLineClass, opts);
    upperClassifier = getClassifier(upperLineCodes, upperLineClass, opts);

    save(classifierFilePath, 'lowerClassifier', 'upperClassifier');
    fprintf('%d: Saving %s\n', opts.seed, classifierFilePath)
  else
    load(classifierFilePath);
    fprintf('%d: Loading %s\n', opts.seed, classifierFilePath)
  end
  fprintf('%d: Mean train confusion matrix for lower classifier is:\n%g\t%g\n%g\t%g\n', ...
    opts.seed, lowerClassifier.meanConfusionMatTrain)
  fprintf('%d: Mean test confusion matrix for lower classifier is:\n%g\t%g\n%g\t%g\n', ...
    opts.seed, lowerClassifier.meanConfusionMatTest)
  fprintf('%d: Mean train confusion matrix for upper classifier is:\n%g\t%g\n%g\t%g\n', ...
    opts.seed, upperClassifier.meanConfusionMatTrain)
  fprintf('%d: Mean test confusion matrix for upper classifier is:\n%g\t%g\n%g\t%g\n', ...
    opts.seed, upperClassifier.meanConfusionMatTest)

  % Generating diameter plot for testing set (only for best descriptor)
  testResultsFilePath = fullfile(expSeedDir, 'testResults.mat');
  if ~exist(testResultsFilePath)
    testResults = getTestResults(trainSeed, lowerClassifier, upperClassifier, dirs, opts);

    save(testResultsFilePath, 'testResults');
    fprintf('%d: Saving %s\n', opts.seed, testResultsFilePath)
  else
    load(testResultsFilePath);
    fprintf('%d: Loading %s\n', opts.seed, testResultsFilePath)
  end
end


%--------------------------------------------------
function lines = getHoughLines(roi, opts)
%--------------------------------------------------
  roiGray = rgb2gray(roi);
  roiW = size(roiGray, 2);
  roiH = size(roiGray, 1);

  % edge
  roiEdge = imfilter(roiGray, fspecial('laplacian'));
  roiEdge = imadjust(roiEdge, stretchlim(roiEdge, [0.05, 0.95]), []);

  % hough
  [H, T, R] = hough(roiEdge, 'RhoResolution', 1, 'ThetaResolution', 1);
  if opts.visualize
    figure, imshow(H, [], 'XData', T, 'YData', R, ...
        'InitialMagnification', 'fit');
    xlabel('\theta'), ylabel('\rho');
    axis on, axis normal, hold on;
  end

  % hough peaks
  P  = houghpeaks(H, opts.numberOfPeaks, 'threshold', ceil(opts.threshold*max(H(:))));
  if opts.visualize
    x = T(P(:,2)); y = R(P(:,1));
    plot(x, y, 's', 'color', 'white');
  end

  if true
    lines = houghlines(roiEdge, T, R, P, 'FillGap', 5, 'MinLength', opts.minLength*roiW);
  else
    lines = repmat(struct(), 1, size(P,1));
    for peak=1:size(P,1)
      theta = T(P(peak,2));
      thetaRad = deg2rad(theta);
      rho = R(P(peak,1));

      lines(peak).theta = theta;
      lines(peak).rho = rho;
      if sin(thetaRad) ~= 0
        lines(peak).point1 = [0, (rho-0*cos(thetaRad))/sin(thetaRad)] + 1;
        lines(peak).point2 = [roiW-1, (rho-(roiW-1)*cos(thetaRad))/sin(thetaRad)] + 1;
      else
        lines(peak).point1 = [0, 0] + 1;
        lines(peak).point2 = [roiW-1, roiH-1] + 1;
      end
    end
  end
  
  if opts.visualize
    figure, imshow(roi);
    for k = 1:length(lines)
      xy = [lines(k).point1; lines(k).point2];
      hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'yellow');

      % Plot beginnings and ends of lines
      hold on; plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
      hold on; plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    end
  end
end


%--------------------------------------------------
function [posLowerLine, posUpperLine, negLowerLine, negUpperLine] = ...
    matchLines(roi, gtLowerLine, gtUpperLine, houghLines, opts)
%--------------------------------------------------
  % upper:  AA   A-B BB
  %   EA=EC=E        F=FB=FD
  % lower:  CC C-D   DD
  if gtLowerLine.point1(1) < gtLowerLine.point2(1)
    C = gtLowerLine.point1;
    D = gtLowerLine.point2;
  else
    C = gtLowerLine.point2;
    D = gtLowerLine.point1;
  end
  if gtUpperLine.point1(1) < gtUpperLine.point2(1)
    A = gtUpperLine.point1;
    B = gtUpperLine.point2;
  else
    A = gtUpperLine.point2;
    B = gtUpperLine.point1;
  end

  % compute houghLines distance from gtLowerLine and gtUpperLine
  lowerDist = zeros(1, length(houghLines));
  upperDist = zeros(1, length(houghLines));
  for hL=1:length(houghLines)
    houghLine = houghLines(hL);
    if houghLine.point1(1) < houghLine.point2(1)
      E = houghLine.point1;
      F = houghLine.point2;
    else
      E = houghLine.point2;
      F = houghLine.point1;
    end

    unionABEF = union(A(1):B(1), E(1):F(1));
    intersectABEF = intersect(A(1):B(1), E(1):F(1));
    AA(1) = unionABEF(1);
    AA(2) = A(2) - (A(1)-AA(1))*(B(2)-A(2))/(B(1)-A(1));
    BB(1) = unionABEF(end);
    BB(2) = B(2) + (B(2)-A(2))*(BB(1)-B(1))/(B(1)-A(1));
    EA(1) = unionABEF(1);
    EA(2) = E(2) - (E(1)-EA(1))*(F(2)-E(2))/(F(1)-E(1));
    FB(1) = unionABEF(end);
    FB(2) = F(2) + (F(2)-E(2))*(FB(1)-F(1))/(F(1)-E(1));
    unionCDEF = union(C(1):D(1), E(1):F(1));
    intersectCDEF = intersect(C(1):D(1), E(1):F(1));
    CC(1) = unionCDEF(1);
    CC(2) = C(2) - (C(1)-CC(1))*(D(2)-C(2))/(D(1)-C(1));
    DD(1) = unionCDEF(end);
    DD(2) = D(2) + (D(2)-C(2))*(DD(1)-D(1))/(D(1)-C(1));
    EC(1) = unionCDEF(1);
    EC(2) = E(2) - (E(1)-EC(1))*(F(2)-E(2))/(F(1)-E(1));
    FD(1) = unionCDEF(end);
    FD(2) = F(2) + (F(2)-E(2))*(FD(1)-F(1))/(F(1)-E(1));

    if length(intersectABEF) < opts.intersectionRatio*min(B(1)-A(1),F(1)-E(1))
      upperDist(hL) = inf;
    elseif (EA(2)-AA(2))*(FB(2)-BB(2)) >= 0
      % AA-BB
      % EA-FB
      upperDist(hL) = (abs(EA(2)-AA(2))+abs(FB(2)-BB(2)))/2;
    else
      % AA FB
      %   X
      % EA BB
      upperDist(hL) = (abs(EA(2)-AA(2))+abs(FB(2)-BB(2)))/4;
    end
    
    if length(intersectCDEF) < opts.intersectionRatio*min(D(1)-C(1),F(1)-E(1))
      lowerDist(hL) = inf;
    elseif (EC(2)-CC(2))*(FD(2)-DD(2)) >= 0
      % CC-DD
      % EC-FD
      lowerDist(hL) = (abs(EC(2)-CC(2))+abs(FD(2)-DD(2)))/2;
    else
      % CC FD
      %   X
      % EC DD
      lowerDist(hL) = (abs(EC(2)-CC(2))+abs(FD(2)-DD(2)))/4;
    end
  end

  posLowerLine = struct('point1', {}, 'point2', {}, 'value', {});
  for hL = 1:length(houghLines)
    if lowerDist(hL) < opts.maxDifference && ~isinf(lowerDist(hL))
      houghLine = houghLines(hL);
      if houghLine.point1(1) > houghLine.point2(1)
        temp = houghLine.point1;
        houghLine.point1 = houghLine.point2;
        houghLine.point2 = temp;
      end
      posLowerLine(end+1).point1 = houghLine.point1;
      posLowerLine(end).point2 = houghLine.point2;
      posLowerLine(end).value = lowerDist(hL);
    end
  end
  negLowerLine = struct('point1', {}, 'point2', {}, 'value', {});
  for hL = 1:length(houghLines)
    if lowerDist(hL) > opts.maxDifference && ~isinf(lowerDist(hL))
      houghLine = houghLines(hL);
      if houghLine.point1(1) > houghLine.point2(1)
        temp = houghLine.point1;
        houghLine.point1 = houghLine.point2;
        houghLine.point2 = temp;
      end
      negLowerLine(end+1).point1 = houghLine.point1;
      negLowerLine(end).point2 = houghLine.point2;
      negLowerLine(end).value = lowerDist(hL);
    end
  end
  if opts.equalPosNegCount && length(negLowerLine) > length(posLowerLine)
    rng(opts.seed); % take as much negatives as positives
    negLowerLine = negLowerLine(randsample(1:length(negLowerLine), length(posLowerLine)));
  end
  
  posUpperLine = struct('point1', {}, 'point2', {}, 'value', {});
  for hL = 1:length(houghLines)
    if upperDist(hL) < opts.maxDifference && ~isinf(upperDist(hL))
      houghLine = houghLines(hL);
      if houghLine.point1(1) > houghLine.point2(1)
        temp = houghLine.point1;
        houghLine.point1 = houghLine.point2;
        houghLine.point2 = temp;
      end
      posUpperLine(end+1).point1 = houghLine.point1;
      posUpperLine(end).point2 = houghLine.point2;
      posUpperLine(end).value = upperDist(hL);
    end
  end
  negUpperLine = struct('point1', {}, 'point2', {}, 'value', {});
  for hL = 1:length(houghLines)
    if upperDist(hL) > opts.maxDifference && ~isinf(upperDist(hL))
      houghLine = houghLines(hL);
      if houghLine.point1(1) > houghLine.point2(1)
        temp = houghLine.point1;
        houghLine.point1 = houghLine.point2;
        houghLine.point2 = temp;
      end
      negUpperLine(end+1).point1 = houghLine.point1;
      negUpperLine(end).point2 = houghLine.point2;
      negUpperLine(end).value = upperDist(hL);
    end
  end
  if opts.equalPosNegCount && length(negUpperLine) > length(posUpperLine)
    rng(opts.seed); % take as much negatives as positives
    negUpperLine = negUpperLine(randsample(1:length(negUpperLine), length(posUpperLine)));
  end
  
  if opts.visualize
    figure, imshow(roi);
    for gtLine = [gtLowerLine, gtUpperLine]
      xy = [gtLine.point1; gtLine.point2];
      hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');

      % Plot beginnings and ends of lines
      hold on; plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'green');
      hold on; plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'green');
    end
    for houghLine = [negLowerLine, negUpperLine]
      xy = [houghLine.point1; houghLine.point2];
      hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');

      % Plot beginnings and ends of lines
      hold on; plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'red');
      hold on; plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
    end
    for houghLine = [posLowerLine, posUpperLine]
      xy = [houghLine.point1; houghLine.point2];
      hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'yellow');

      % Plot beginnings and ends of lines
      hold on; plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
      hold on; plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    end
  end
end


%--------------------------------------------------
function code = getLineDescriptor(roi, line, opts)
%--------------------------------------------------
  roiGray = rgb2gray(roi);

  switch opts.lineDescriptor
    case 'ProfilePlot'
      profilePlot = improfile(roiGray, [line.point1(1), line.point2(1)], [line.point1(2), line.point2(2)]);
      
      % to neutralize bright vertical line
      h = 5;
      profilePlotAfterMedianFilter = zeros(1, length(profilePlot)-2*h);
      for pP = h+1:length(profilePlot)-h
        profilePlotAfterMedianFilter(pP-h) = median(profilePlot(pP-h:pP+h));
      end

      % same length
      code = zeros(1, opts.profilePlotLength);
      for c = 1:opts.profilePlotLength
        code(c) = profilePlotAfterMedianFilter(ceil(c*length(profilePlotAfterMedianFilter)/opts.profilePlotLength));
      end

      % normalize
      meanCode = mean(code);
      stdCode = std(code);
      code = (code - meanCode) / stdCode;
    case 'TinyImage'
      % A-B
      % C-D
      A = [0, 0]; B = [opts.tinyImageWidth, 0];
      C = [A(1), opts.tinyImageHeight]; D = [B(1), opts.tinyImageHeight];

      L = line.point1; R = line.point2;
      theta = 90; M = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
      v = (R-L) / norm(R-L) * M * opts.tinyImageLineHeight/2;
      AP = L + v; BP = R + v;
      CP = L - v; DP = R - v;

      tform = estimateGeometricTransform([AP; BP; CP; DP], [A; B; C; D], 'affine');

      outputView = imref2d([opts.tinyImageHeight, opts.tinyImageWidth], [1, opts.tinyImageWidth], [1, opts.tinyImageHeight]);
      roiWarp = imwarp(roiGray, tform, 'OutputView', outputView);
      roiWarp = double(roiWarp);
      roiWarp = (roiWarp - mean(roiWarp(:))) / std(roiWarp(:));

      roiWarpT = roiWarp';
      codeBefore = roiWarpT(:);

      % to neutralize bright vertical line
      h = 5;
      code = zeros(1, length(codeBefore)-2*h);
      for pP = h+1:length(codeBefore)-h
        code(pP-h) = median(codeBefore(pP-h:pP+h));
      end
      
      if opts.visualize
        figure;
        subplot(1, 3, 1); imshow(roiGray); hold on; plot([L(1), R(1)], [L(2), R(2)], 'LineWidth', 1, 'Color', 'yellow');
        subplot(1, 3, 2); imagesc(roiWarp);
        subplot(1, 3, 3); plot(codeBefore); hold on; plot(h+1:h+length(code), code, 'LineWidth', 1);
      end
    case 'BriefLike'
      % A-B
      % C-D
      A = [0, 0]; B = [opts.briefLikeWidth, 0];
      C = [A(1), opts.briefLikeHeight]; D = [B(1), opts.briefLikeHeight];

      L = line.point1; R = line.point2;
      theta = 90; M = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
      v = (R-L) / norm(R-L) * M * opts.briefLikeHeight/2;
      AP = L + v; BP = R + v;
      CP = L - v; DP = R - v;

      tform = estimateGeometricTransform([A; B; C; D], [AP; BP; CP; DP], 'affine');

      [x1, y1] = transformPointsForward(tform, opts.desc.point1(:, 1), opts.desc.point1(:, 2));
      [x2, y2] = transformPointsForward(tform, opts.desc.point2(:, 1), opts.desc.point2(:, 2));

      values1 = impixel(roiGray, ceil(x1), ceil(y1)); values1 = values1(: ,1);
      values2 = impixel(roiGray, ceil(x2), ceil(y2)); values2 = values2(: ,1);

      code = values1 - values2 > 0;
      % values of pairs outside the image are random
      code(isnan(values1) | isnan(values2)) = randi(2, sum(isnan(values1) | isnan(values2)), 1) - 1;

      if opts.visualize
        figure; imshow(roiGray);
        for p = 1:length(x1)
          hold on; plot([x1(p), x2(p)], [y1(p), y2(p)], 'LineWidth', 1, 'Color', 'yellow');
        end
      end
  end
end


%--------------------------------------------------
function desc = getBriefLikeDescriptor(opts)
%--------------------------------------------------
  desc.point1 = normrnd(0, opts.briefLikeSigma, opts.briefLikePairs, 2);
  desc.point2 = normrnd(0, opts.briefLikeSigma, opts.briefLikePairs, 2);

  % leave only pairs which are inside the rectangle
  isInside = max(abs(desc.point1), [], 2) < 1 & max(abs(desc.point2), [], 2) < 1;
  desc.point1 = desc.point1(isInside, :);
  desc.point2 = desc.point2(isInside, :);

  desc.point1(:, 1) = desc.point1(:, 1) * opts.briefLikeWidth/2 + opts.briefLikeWidth/2;
  desc.point1(:, 2) = desc.point1(:, 2) * opts.briefLikeHeight/2 + opts.briefLikeHeight/2;
  desc.point2(:, 1) = desc.point2(:, 1) * opts.briefLikeWidth/2 + opts.briefLikeWidth/2;
  desc.point2(:, 2) = desc.point2(:, 2) * opts.briefLikeHeight/2 + opts.briefLikeHeight/2;

  if opts.visualize
    figure; imshow(ones(opts.briefLikeHeight, opts.briefLikeWidth));
    for p = 1:size(desc.point1, 1)
      hold on; plot([desc.point1(p, 1); desc.point2(p, 1)], ...
        [desc.point1(p, 2); desc.point2(p, 2)], 'LineWidth', 1, 'Color', 'black');
    end
  end
end

%--------------------------------------------------
function classifier = getClassifier(lineCodes, lineClass, opts)
%--------------------------------------------------
  classifier.kFold = 5;
  classifier.cv = cvpartition(lineClass, 'kfold', classifier.kFold);

  classifier.meanConfusionMatTrain = [0, 0; 0, 0];
  classifier.meanConfusionMatTest = [0, 0; 0, 0];

  for k=1:classifier.kFold
    classifier.parts{k}.isTrain = training(classifier.cv, k);
    classifier.parts{k}.isTest = test(classifier.cv, k);

    rng(opts.seed);
    t = templateTree('MinLeafSize', opts.minLeafSize);
    rusTree = fitensemble(lineCodes(classifier.parts{k}.isTrain, :), ...
      lineClass(classifier.parts{k}.isTrain), ...
      'RUSBoost', opts.numberOfTrees, t, ...
      'LearnRate', opts.learnRate);

    % train statistics
    classifier.parts{k}.fitLineClassTrain = predict(rusTree, ...
      lineCodes(classifier.parts{k}.isTrain, :));

    tab = tabulate(lineClass(classifier.parts{k}.isTrain));
    classifier.parts{k}.confusionMatTrain = ...
      bsxfun(@rdivide, confusionmat(lineClass(classifier.parts{k}.isTrain), ...
      classifier.parts{k}.fitLineClassTrain), tab(:,2)) * 100;

    classifier.meanConfusionMatTrain = classifier.meanConfusionMatTrain + ...
      classifier.parts{k}.confusionMatTrain;

    classifier.parts{k}.lossTrain0 = loss(rusTree, ...
      lineCodes(classifier.parts{k}.isTrain & lineClass==0, :), lineClass(classifier.parts{k}.isTrain & lineClass==0), ...
      'mode', 'cumulative');
    classifier.parts{k}.lossTrain1 = loss(rusTree, ...
      lineCodes(classifier.parts{k}.isTrain & lineClass==1, :), lineClass(classifier.parts{k}.isTrain & lineClass==1), ...
      'mode', 'cumulative');

    % test statistics
    classifier.parts{k}.fitLineClassTest = predict(rusTree, ...
      lineCodes(classifier.parts{k}.isTest, :));

    tab = tabulate(lineClass(classifier.parts{k}.isTest));
    classifier.parts{k}.confusionMatTest = ...
      bsxfun(@rdivide, confusionmat(lineClass(classifier.parts{k}.isTest), ...
      classifier.parts{k}.fitLineClassTest), tab(:,2)) * 100;

    classifier.meanConfusionMatTest = classifier.meanConfusionMatTest + ...
      classifier.parts{k}.confusionMatTest;

    classifier.parts{k}.lossTest0 = loss(rusTree, ...
      lineCodes(classifier.parts{k}.isTest & lineClass==0, :), lineClass(classifier.parts{k}.isTest & lineClass==0), ...
      'mode', 'cumulative');
    classifier.parts{k}.lossTest1 = loss(rusTree, ...
      lineCodes(classifier.parts{k}.isTest & lineClass==1, :), lineClass(classifier.parts{k}.isTest & lineClass==1), ...
      'mode', 'cumulative');

    fprintf('%d: Classification fold number %d has been computed\n', opts.seed, k)
    if opts.visualize
      figure;
      plot(classifier.parts{k}.loss);
      xlabel('Number of trees');
      ylabel(['Cross-validated error - fold', num2str(k)]);
      legend('RUSBoost','Location','NE');
    end
  end

  classifier.meanConfusionMatTrain = classifier.meanConfusionMatTrain / classifier.kFold;
  classifier.meanConfusionMatTest = classifier.meanConfusionMatTest / classifier.kFold;

  classifier.rusTree = fitensemble(lineCodes, lineClass, ...
      'RUSBoost', opts.numberOfTrees, t, ...
      'LearnRate', opts.learnRate);

  if opts.visualize
    figure;
    for p = 1:classifier.kFold
      subplot(ceil(classifier.kFold / 2), 2, p);
      plot(lowerClassifier.parts{p}.lossTrain0, 'LineWidth', 1, 'Color', 'red');
      hold on; plot(lowerClassifier.parts{p}.lossTrain1, 'LineWidth', 1, 'Color', 'green');
      plot(lowerClassifier.parts{p}.lossTest0, ':', 'LineWidth', 1, 'Color', 'red');
      hold on; plot(lowerClassifier.parts{p}.lossTest1, ':', 'LineWidth', 1, 'Color', 'green');
    end
  end
end


%--------------------------------------------------
function testResults = getTestResults(trainSeed, lowerClassifier, upperClassifier, dirs, opts)
%--------------------------------------------------
  testResults = {};

  for tI = 1:length(trainSeed.testSetIndices)
    testIndex = trainSeed.testSetIndices(tI);
    load(fullfile(dirs.roiDir, ['rois-', num2str(testIndex), '.mat']));

    testResult = repmat(struct('bestLowerLine', {}, 'bestUpperLine', {}, ...
      'gtDistance', {}, 'bestDistance', {}), 1, length(roiInfos));
    for rI = 1:length(roiInfos)
      tic
      roiInfo = roiInfos(rI);
      roi = roiInfo.roi;
      gtLowerLine = roiInfo.gtLowerLine;
      gtUpperLine = roiInfo.gtUpperLine;

      houghLines = getHoughLines(roi, opts);

      lineScores = zeros(length(houghLines), 2);

      for hL = 1:length(houghLines)
        houghLine = houghLines(hL);
        houghLines(hL).centerY = (houghLine.point1(2) + houghLine.point2(2)) / 2;
      end
      lowerHoughLines = houghLines(cat(1, houghLines.centerY) > size(roi, 1) / 2);
      upperHoughLines = houghLines(cat(1, houghLines.centerY) < size(roi, 1) / 2);

      lowerLineScores = zeros(1, length(lowerHoughLines));
      for hL = 1:length(lowerHoughLines)
        houghLine = lowerHoughLines(hL);
        lineCode = getLineDescriptor(roi, houghLine, opts);
        [~, score] = predict(lowerClassifier.rusTree, lineCode, 'learners', 1:opts.testLearners);
        lowerLineScores(hL) = score(2);
      end

      upperLineScores = zeros(1, length(upperHoughLines));
      for hL = 1:length(upperHoughLines)
        houghLine = upperHoughLines(hL);
        lineCode = getLineDescriptor(roi, houghLine, opts);
        [~, score] = predict(upperClassifier.rusTree, lineCode, 'learners', 1:opts.testLearners);
        upperLineScores(hL) = score(2);
      end

      [~, lMaxIndex] = max(lowerLineScores);
      bestLowerLine = lowerHoughLines(lMaxIndex);

      [~, uMaxIndex] = max(upperLineScores);
      bestUpperLine = upperHoughLines(uMaxIndex);

      gtMeanDist = getMeanDistance(gtLowerLine.point1, gtLowerLine.point2, ...
        gtUpperLine.point1, gtUpperLine.point2);
      bestMeanDist = getMeanDistance(bestLowerLine.point1, bestLowerLine.point2, ...
        bestUpperLine.point1, bestUpperLine.point2);

      testResult(rI).bestLowerLine = bestLowerLine;
      testResult(rI).bestUpperLine = bestUpperLine;
      testResult(rI).gtDistance = gtMeanDist;
      testResult(rI).bestDistance = bestMeanDist;

      if opts.visualize
        figure, imshow(roi);
        for line = [gtLowerLine, gtUpperLine]
          xy = [line.point1; line.point2];
          hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
        end
        for line = [bestLowerLine, bestUpperLine]
          xy = [line.point1; line.point2];
          hold on; plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'yellow');
        end
      end
      toc
    end

    testResults{tI}.testIndex = testIndex;
    testResults{tI}.testResult = testResult;
  end
end


%--------------------------------------------------
function meanDist = getMeanDistance(A, B, C, D)
%--------------------------------------------------
  dists = zeros(1, 4);
  dists(1) = getDistance(C, D, A);
  dists(2) = getDistance(C, D, B);
  dists(3) = getDistance(A, B, C);
  dists(4) = getDistance(A, B, D);

  meanDist = mean(dists);
end


%--------------------------------------------------
function dist = getDistance(P1, P2, P0)
%--------------------------------------------------
  x1 = P1(1); y1 = P1(2);
  x2 = P2(1); y2 = P2(2);
  x0 = P0(1); y0 = P0(2);
  dist = abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/sqrt((y2-y1)^2+(x2-x1)^2);
end
