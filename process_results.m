%--------------------------------------------------
function process_results
%--------------------------------------------------
  opts.visualize = false;
  opts.slicesNumber = 10;
  opts.lineDescriptor = 'TinyImage';
  
  dirs.anonymousDir = 'data/anonymous';
  dirs.expDir = fullfile(dirs.anonymousDir, ['exp', opts.lineDescriptor]);
  dirs.roiDir = fullfile(dirs.anonymousDir, 'roi');
  dirs.imdbDir = fullfile(dirs.anonymousDir, 'imdb');

  if opts.visualize
    close all;
  end

  % add frame indexes to testRestults
  if false
    for seed = 1:opts.slicesNumber
      expSeedDir = fullfile(dirs.expDir, ['seed-', num2str(seed)]);
      testResultsFilePath = fullfile(expSeedDir, 'testResults.mat');
      imdbFilePath = fullfile(dirs.imdbDir, ['imdb-', num2str(seed), '.mat']);

      load(testResultsFilePath);
      load(imdbFilePath);

      for tR = 1:length(testResults)
        testIndex = trainSeed.testSetIndices(tR);

        load(fullfile(dirs.roiDir, ['rois-', num2str(testIndex), '.mat']));
        frameIndices = cat(1, roiInfos.frameIndex)';

        for fI = 1:length(frameIndices)
          testResults{tR}.testResult(fI).frameIndex = frameIndices(fI);
        end
      end

      save(testResultsFilePath, 'testResults');
    end
    
    return;
  end
  
  gtFmdAll = [];
  bestFmdAll = [];
  gtMinMaxAll = [];
  gtMeanAll = [];
  gtStdAll = [];

  for seed = 1:opts.slicesNumber
    expSeedDir = fullfile(dirs.expDir, ['seed-', num2str(seed)]);
    testResultsFilePath = fullfile(expSeedDir, 'testResults.mat');
    imdbFilePath = fullfile(dirs.imdbDir, ['imdb-', num2str(seed), '.mat']);

    load(testResultsFilePath);
    load(imdbFilePath);

    gtFmd = [];
    bestFmd = [];
    gtMinMax = [];
    gtMean = [];
    gtStd = [];
    for tR = 1:length(testResults)
      testIndex = trainSeed.testSetIndices(tR);

      frameIndices = cat(1, testResults{tR}.testResult.frameIndex)';

      x = frameIndices(1:10:length(frameIndices));
      testResult = testResults{tR}.testResult(1:10:length(frameIndices));

      gtY = cat(1, testResult.gtDistance)';
      gtYMF = medianFilter(gtY);
      %gtCoeffs = polyfit(x, gtYMF, 7);
      %gtYMF = polyval(gtCoeffs, x);

      bestY = cat(1, testResult.bestDistance)';
      bestYMF = medianFilter(bestY);
      %bestCoeffs = polyfit(x, bestYMF, 7);
      %bestYMF = polyval(bestCoeffs, x);

      gtFmd(end + 1) = (max(gtYMF) - min(gtYMF)) / min(gtYMF);
      bestFmd(end + 1) = (max(bestYMF) - min(bestYMF)) / min(bestYMF);
      gtMinMax(end + 1) = max(gtYMF) - min(gtYMF);
      gtMean(end + 1) = mean(gtYMF);
      gtStd(end + 1) = std(gtYMF);

      if opts.visualize
        figure;
        plot(x, gtY/30, '--', 'Color', 'green'); hold on;
        plot(x, gtYMF/30, 'Color', 'green', 'LineWidth', 1);
        plot(x, bestY/30, '--', 'Color', 'black');
        plot(x, bestYMF/30, 'Color', 'black', 'LineWidth', 1);
        xlabel('frame index');
        ylabel('artery diameter [mm]');
        title(strcat('diff(gtFMD,autoFMD)=',  num2str((gtFmd(end) - bestFmd(end))*100), '%'));
        set(gca, 'FontSize', 14)
      end
    end

    gtFmdAll = [gtFmdAll, gtFmd];
    bestFmdAll = [bestFmdAll, bestFmd];
    gtMinMaxAll = [gtMinMaxAll, gtMinMax];
    gtMeanAll = [gtMeanAll, gtMean];
    gtStdAll = [gtStdAll, gtStd];
  end

  gtFmdAll = gtFmdAll * 100;
  bestFmdAll = bestFmdAll * 100;

  absGtBestFmdAll = abs(gtFmdAll-bestFmdAll);
  
  q1 = prctile(absGtBestFmdAll, 25);
  q3 = prctile(absGtBestFmdAll, 75);
  isOutlier = absGtBestFmdAll > q3+1.5*(q3-q1) | absGtBestFmdAll < q1-1.5*(q3-q1);
  isOutlier = bestFmdAll > 20;

  figure;
  bp = boxplot(absGtBestFmdAll, 'labels', '');
  title(strcat('abs(gtFMD''-autoFMD'')=', num2str(mean(absGtBestFmdAll(~isOutlier))), '\pm', num2str(std(absGtBestFmdAll(~isOutlier))), '%'));
  ylabel('abs(gtFMD-autoFMD)[%]');
  set(gca, 'XTickLabel', {' '});
  set(gca, 'FontSize', 14)

  cc = corrcoef(gtFmdAll(~isOutlier), bestFmdAll(~isOutlier));
  figure;
  plot(gtFmdAll, bestFmdAll, 'o'); hold on;
  plot([0, max(gtFmdAll)], [0, max(gtFmdAll)], 'Color', 'green');
  title(strcat('corr(gtFMD'', autoFMD'')=', num2str(cc(1, 2))));
  xlabel('gtFMD[%]');
  ylabel('autoFMD[%]');
  coeffs = polyfit(gtFmdAll, bestFmdAll, 1);
  fitBestFmdAll = polyval(coeffs, [0, max(gtFmdAll)]);
  plot([0, max(gtFmdAll)], fitBestFmdAll, '--', 'Color', 'red');
  coeffsNoOut = polyfit(gtFmdAll(~isOutlier), bestFmdAll(~isOutlier), 1);
  fitBestFmdAllNoOut = polyval(coeffsNoOut, [0, max(gtFmdAll)]);
  plot([0, max(gtFmdAll)], fitBestFmdAllNoOut, 'Color', 'red');
  set(gca, 'FontSize', 14)

  figure;
  plot(gtMinMaxAll/30, absGtBestFmdAll, 'o'); hold on;
  xlabel('max(gt)-min(gt)[mm]');
  ylabel('abs(gtFMD-autoFMD)[%]');
  set(gca, 'FontSize', 14)

  figure;
  plot(gtMeanAll/30, absGtBestFmdAll, 'o'); hold on;
  xlabel('mean(gt)[mm]');
  ylabel('abs(gtFMD-autoFMD)[%]');
  set(gca, 'FontSize', 14)

  figure;
  plot(gtStdAll/30, absGtBestFmdAll, 'o'); hold on;
  xlabel('std(gt)[mm]');
  ylabel('abs(gtFMD-autoFMD)[%]');
  set(gca, 'FontSize', 14)
end


%--------------------------------------------------
function xMF = medianFilter(x)
%--------------------------------------------------
  h = 10;
  xMF = x;
  for pP = h+1:length(x)-h
    xMF(pP) = median(x(pP-h:pP+h));
  end
  xMF(1:h) = xMF(h+1);
  xMF(end-h+1:end) = xMF(end-h);
end
