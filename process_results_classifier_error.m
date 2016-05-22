%--------------------------------------------------
function process_results_classifier_error
%--------------------------------------------------

load('data/anonymous/expTinyImage/seed-1/classifier.mat') % you can change seed number

lossTrain0 = zeros(500, 1);
lossTest0 = zeros(500, 1);
lossTrain1 = zeros(500, 1);
lossTest1 = zeros(500, 1);
tn = [];
tp = [];
fn = [];
fp = [];

for p = 1:5
  t = lowerClassifier.parts{p}.confusionMatTrain; % you can also use upperClassifier
  tn = [tn, t(1, 1)];
  tp = [tp, t(2, 2)];
  fn = [fn, t(1, 2)];
  fp = [fp, t(2, 1)];

  lossTrain0 = lossTrain0 + lowerClassifier.parts{p}.lossTrain0;
  lossTest0 = lossTest0 + lowerClassifier.parts{p}.lossTest0;
  lossTrain1 = lossTrain0 + lowerClassifier.parts{p}.lossTrain1;
  lossTest1 = lossTest1 + lowerClassifier.parts{p}.lossTest1;
end

lossTrain0 = lossTrain0 / 5;
lossTest0 = lossTest0 / 5;
lossTrain1 = lossTrain1 / 5;
lossTest1 = lossTest1 / 5;

plot(lossTrain0, '--', 'LineWidth', 1, 'Color', 'red');
hold on; plot(lossTest0, 'LineWidth', 1, 'Color', 'red');
hold on; plot(lossTrain1, '--', 'LineWidth', 1, 'Color', 'green');
hold on; plot(lossTest1, 'LineWidth', 1, 'Color', 'green');
set(gca, 'FontSize', 14);
xlabel('number of trees');
ylabel('error');
