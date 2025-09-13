/**
 * XGBoost Usage Examples
 * This file demonstrates how to use the XGBoost algorithm for gradient boosting
 */

import XGBoost from '../lib/xgboost.js';

// Sample dataset for demonstration
const sampleData = [
  { color: 'red', shape: 'circle', size: 'small', liked: true },
  { color: 'blue', shape: 'square', size: 'medium', liked: false },
  { color: 'green', shape: 'triangle', size: 'large', liked: true },
  { color: 'red', shape: 'square', size: 'small', liked: false },
  { color: 'blue', shape: 'circle', size: 'medium', liked: true },
  { color: 'green', shape: 'hexagon', size: 'large', liked: false },
  { color: 'yellow', shape: 'circle', size: 'small', liked: true },
  { color: 'purple', shape: 'square', size: 'medium', liked: false }
];

const features = ['color', 'shape', 'size'];
const target = 'liked';

console.log('=== XGBoost Basic Usage ===');

// Basic XGBoost usage
const xgb = new XGBoost(target, features);
xgb.train(sampleData);

console.log('Training completed!');
console.log('Number of trees:', xgb.getTreeCount());
console.log('Best iteration:', xgb.getBestIteration());

// Make predictions
const testSample = { color: 'blue', shape: 'hexagon', size: 'medium' };
const prediction = xgb.predict(testSample);
console.log('Prediction for test sample:', prediction);

// Evaluate accuracy
const accuracy = xgb.evaluate(sampleData);
console.log('Training accuracy:', accuracy.toFixed(3));

console.log('\n=== XGBoost Configuration ===');

// XGBoost with custom configuration
const config = {
  nEstimators: 50,
  learningRate: 0.1,
  maxDepth: 4,
  minChildWeight: 2,
  subsample: 0.8,
  colsampleByTree: 0.8,
  regAlpha: 0.1,
  regLambda: 1.0,
  objective: 'binary',
  earlyStoppingRounds: 10,
  validationFraction: 0.2,
  randomState: 42
};

const xgbCustom = new XGBoost(target, features, config);
xgbCustom.train(sampleData);

console.log('Custom configuration training completed!');
console.log('Number of trees:', xgbCustom.getTreeCount());
console.log('Best iteration:', xgbCustom.getBestIteration());

// Get feature importance
const importance = xgbCustom.getFeatureImportance();
console.log('Feature importance:', importance);

// Get boosting history
const history = xgbCustom.getBoostingHistory();
console.log('Training loss progression:', history.trainLoss.slice(0, 5));
console.log('Validation loss progression:', history.validationLoss.slice(0, 5));

console.log('\n=== XGBoost Model Persistence ===');

// Export model
const modelJson = xgbCustom.toJSON();
console.log('Model exported successfully');
console.log('Model contains', modelJson.trees.length, 'trees');

// Import model
const xgbImported = new XGBoost(modelJson);
console.log('Model imported successfully');
console.log('Imported model tree count:', xgbImported.getTreeCount());

// Verify predictions are the same
const originalPrediction = xgbCustom.predict(testSample);
const importedPrediction = xgbImported.predict(testSample);
console.log('Original prediction:', originalPrediction);
console.log('Imported prediction:', importedPrediction);
console.log('Predictions match:', originalPrediction === importedPrediction);

console.log('\n=== XGBoost Different Objectives ===');

// Regression example
const regressionData = [
  { feature1: 1, feature2: 2, target: 10 },
  { feature1: 2, feature2: 3, target: 20 },
  { feature1: 3, feature2: 4, target: 30 },
  { feature1: 4, feature2: 5, target: 40 }
];

const regressionConfig = {
  nEstimators: 20,
  learningRate: 0.1,
  objective: 'regression',
  randomState: 42
};

const xgbRegression = new XGBoost('target', ['feature1', 'feature2'], regressionConfig);
xgbRegression.train(regressionData);

const regressionPrediction = xgbRegression.predict({ feature1: 5, feature2: 6 });
console.log('Regression prediction:', regressionPrediction);

console.log('\n=== XGBoost Early Stopping ===');

// Early stopping example
const earlyStoppingConfig = {
  nEstimators: 100,
  learningRate: 0.1,
  earlyStoppingRounds: 5,
  validationFraction: 0.3,
  randomState: 42
};

const xgbEarlyStop = new XGBoost(target, features, earlyStoppingConfig);
xgbEarlyStop.train(sampleData);

console.log('Early stopping training completed!');
console.log('Number of trees:', xgbEarlyStop.getTreeCount());
console.log('Best iteration:', xgbEarlyStop.getBestIteration());

console.log('\n=== XGBoost Performance Comparison ===');

// Compare different algorithms
const algorithms = [
  { name: 'XGBoost (50 trees)', config: { nEstimators: 50, randomState: 42 } },
  { name: 'XGBoost (100 trees)', config: { nEstimators: 100, randomState: 42 } },
  { name: 'XGBoost (200 trees)', config: { nEstimators: 200, randomState: 42 } }
];

algorithms.forEach(alg => {
  const startTime = Date.now();
  const xgb = new XGBoost(target, features, alg.config);
  xgb.train(sampleData);
  const endTime = Date.now();
  
  const accuracy = xgb.evaluate(sampleData);
  console.log(`${alg.name}: ${accuracy.toFixed(3)} accuracy, ${endTime - startTime}ms`);
});

console.log('\n=== XGBoost Feature Selection ===');

// Test different feature selection strategies
const featureConfigs = [
  { name: 'All features', colsampleByTree: 1.0 },
  { name: '80% features', colsampleByTree: 0.8 },
  { name: '60% features', colsampleByTree: 0.6 },
  { name: '40% features', colsampleByTree: 0.4 }
];

featureConfigs.forEach(config => {
  const xgb = new XGBoost(target, features, {
    nEstimators: 30,
    colsampleByTree: config.colsampleByTree,
    randomState: 42
  });
  xgb.train(sampleData);
  
  const accuracy = xgb.evaluate(sampleData);
  console.log(`${config.name}: ${accuracy.toFixed(3)} accuracy`);
});

console.log('\nXGBoost examples completed successfully!');
