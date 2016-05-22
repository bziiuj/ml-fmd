function generate_anonymous_roi()
  opts.generateAnonymousData = true;
  opts.generateFirstFrames = false;

  opts.visualize = false;

  imaDirs = 'data/ima';
  anonymousDir = 'data/anonymous';
  roiDir = fullfile(anonymousDir, 'roi');

  dirInfo = dir(imaDirs);
  dirInfo(~[dirInfo.isdir]) = [];  %remove non-directories
  dirInfo(ismember({dirInfo.name}, {'.', '..'})) = [];  %remove current and parent directory.
  imaFiles = cell(length(dirInfo), 1);
  imaPaths = cell(length(dirInfo), 1);
  for imaIndex = 1:length(dirInfo)
    imaDir = dirInfo(imaIndex).name;
    imaFileInfo = dir(fullfile(imaDirs, imaDir, '*.IMA'));
    imaFiles{imaIndex} = imaFileInfo.name;
    imaPaths{imaIndex} = fullfile(imaDirs, imaDir, imaFiles{imaIndex});
  end

  % print first frames with gtLines
  if opts.generateFirstFrames
    mkdir(firstFramesDir);

    for imaIndex = 1:length(imaPaths)
      imaPath = imaPaths{imaIndex};
      [rois, gtXYs] = getRoisAndXYs(imaPath, 'first', opts);
      print(fullfile(firstFramesDir, [num2str(imaIndex), '.eps']), '-depsc');
      close all;
    end
  end
  
  if opts.generateAnonymousData
    mkdir(roiDir);

    imaFilesNumber = length(dirInfo);
    for imaIndex = 1:imaFilesNumber
      roiInfos = getRoisAndXYs(imaPaths{imaIndex}, 'all', opts);
      save(fullfile(roiDir, ['rois-', num2str(imaIndex), '.mat']), 'roiInfos');
    end
  end
end

%%

function roiInfos = getRoisAndXYs(imaPath, k, opts)
  filePathSer = strcat(imaPath, '.ser2.gt.lines');
  if ~exist(filePathSer)
    filePathSer = strcat(imaPath, '.ser4.gt.lines');
  end

  gtXYs = dlmread(filePathSer, ';');
  roiXY = gtXYs(1, :);
  gtXYs = gtXYs(2:end, :);

  if strcmp(k, 'first')
    gtXYs = gtXYs(1, :);
    k = 1;
    frames = dicomread(imaPath, 'frames', 1:gtXYs(1, 1));
  else
    k = size(gtXYs, 1);
    frames = dicomread(imaPath, 'frames', 'all');
  end

  roiInfos = repmat(struct('frameIndex', [], 'roi', [], 'gtLowerLine', [], 'gtUpperLine', []), 1, k);

  for index=1:k
    frame = frames(:, :, :, gtXYs(index, 1));
    roi = frame(roiXY(2):roiXY(2)+roiXY(4), roiXY(1):roiXY(1)+roiXY(3), :);

    roiInfo.frameIndex = gtXYs(index, 1);
    roiInfo.roi = roi;
    roiInfo.gtLowerLine.point1 = [gtXYs(index, 2), gtXYs(index, 3)];
    roiInfo.gtLowerLine.point2 = [gtXYs(index, 4), gtXYs(index, 5)];
    roiInfo.gtUpperLine.point1 = [gtXYs(index, 6), gtXYs(index, 7)];
    roiInfo.gtUpperLine.point2 = [gtXYs(index, 8), gtXYs(index, 9)];
    roiInfos(index) = roiInfo;

    if opts.visualize
      gtLines = repmat(struct(), 1, 2);
      gtLines(1) = roiInfo.gtLowerLine;
      gtLines(2) = roiInfo.gtUpperLine;

      figure, imshow(roiInfo.roi), hold on
      for k = 1:length(gtLines)
         xy = [gtLines(k).point1; gtLines(k).point2];
         plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

         % Plot beginnings and ends of lines
         plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','green');
         plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','green');
      end
    end
  end
end
