/**
 * XGBoost TypeScript Usage Examples
 * This file demonstrates how to use the XGBoost algorithm with TypeScript
 */

import XGBoost from '../lib/xgboost.js';

// Type definitions for our data
interface SampleData {
  color: string;
  shape: string;
  size: string;
  liked: boolean;
}

interface RegressionData {
  feature1: number;
  feature2: number;
  target: number;
}

// Sample dataset for demonstration
const sampleData: SampleData[] = [
  { color: 'red', shape: 'circle', size: 'small', liked: true },
  { color: 'blue', shape: 'square', size: 'medium', liked: false },
  { color: 'green', shape: 'triangle', size: 'large', liked: true },
  { color: 'red', shape: 'square', size: 'small', liked: false },
  { color: 'blue', shape: 'circle', size: 'medium', liked: true },
  { color: 'green', shape: 'hexagon', size: 'large', liked: false },
  { color: 'yellow', shape: 'circle', size: 'small', liked: true },
  { color: 'purple', shape: 'square', size: 'medium', liked: false }
];

const features: string[] = ['color', 'shape', 'size'];
const target: string = 'liked';

console.log('=== XGBoost TypeScript Basic Usage ===');

// Basic XGBoost usage with type safety
const xgb = new XGBoost(target, features);
xgb.train(sampleData);

console.log('Training completed!');
console.log('Number of trees:', xgb.getTreeCount());
console.log('Best iteration:', xgb.getBestIteration());

// Make predictions with type safety
const testSample: SampleData = { color: 'blue', shape: 'hexagon', size: 'medium' };
const prediction: boolean = xgb.predict(testSample);
console.log('Prediction for test sample:', prediction);

// Evaluate accuracy
const accuracy: number = xgb.evaluate(sampleData);
console.log('Training accuracy:', accuracy.toFixed(3));

console.log('\n=== XGBoost TypeScript Configuration ===');

// XGBoost with custom configuration and type safety
const config = {
  nEstimators: 50,
  learningRate: 0.1,
  maxDepth: 4,
  minChildWeight: 2,
  subsample: 0.8,
  colsampleByTree: 0.8,
  regAlpha: 0.1,
  regLambda: 1.0,
  objective: 'binary' as const,
  earlyStoppingRounds: 10,
  validationFraction: 0.2,
  randomState: 42
};

const xgbCustom = new XGBoost(target, features, config);
xgbCustom.train(sampleData);

console.log('Custom configuration training completed!');
console.log('Number of trees:', xgbCustom.getTreeCount());
console.log('Best iteration:', xgbCustom.getBestIteration());

// Get feature importance with type safety
const importance: { [feature: string]: number } = xgbCustom.getFeatureImportance();
console.log('Feature importance:', importance);

// Get boosting history with type safety
const history = xgbCustom.getBoostingHistory();
console.log('Training loss progression:', history.trainLoss.slice(0, 5));
console.log('Validation loss progression:', history.validationLoss.slice(0, 5));

console.log('\n=== XGBoost TypeScript Model Persistence ===');

// Export model with type safety
const modelJson = xgbCustom.toJSON();
console.log('Model exported successfully');
console.log('Model contains', modelJson.trees.length, 'trees');

// Import model with type safety
const xgbImported = new XGBoost(modelJson);
console.log('Model imported successfully');
console.log('Imported model tree count:', xgbImported.getTreeCount());

// Verify predictions are the same
const originalPrediction: boolean = xgbCustom.predict(testSample);
const importedPrediction: boolean = xgbImported.predict(testSample);
console.log('Original prediction:', originalPrediction);
console.log('Imported prediction:', importedPrediction);
console.log('Predictions match:', originalPrediction === importedPrediction);

console.log('\n=== XGBoost TypeScript Different Objectives ===');

// Regression example with type safety
const regressionData: RegressionData[] = [
  { feature1: 1, feature2: 2, target: 10 },
  { feature1: 2, feature2: 3, target: 20 },
  { feature1: 3, feature2: 4, target: 30 },
  { feature1: 4, feature2: 5, target: 40 }
];

const regressionConfig = {
  nEstimators: 20,
  learningRate: 0.1,
  objective: 'regression' as const,
  randomState: 42
};

const xgbRegression = new XGBoost('target', ['feature1', 'feature2'], regressionConfig);
xgbRegression.train(regressionData);

const regressionPrediction: number = xgbRegression.predict({ feature1: 5, feature2: 6 });
console.log('Regression prediction:', regressionPrediction);

console.log('\n=== XGBoost TypeScript Early Stopping ===');

// Early stopping example with type safety
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

console.log('\n=== XGBoost TypeScript Performance Comparison ===');

// Compare different algorithms with type safety
interface AlgorithmConfig {
  name: string;
  config: {
    nEstimators: number;
    randomState: number;
  };
}

const algorithms: AlgorithmConfig[] = [
  { name: 'XGBoost (50 trees)', config: { nEstimators: 50, randomState: 42 } },
  { name: 'XGBoost (100 trees)', config: { nEstimators: 100, randomState: 42 } },
  { name: 'XGBoost (200 trees)', config: { nEstimators: 200, randomState: 42 } }
];

algorithms.forEach((alg: AlgorithmConfig) => {
  const startTime: number = Date.now();
  const xgb = new XGBoost(target, features, alg.config);
  xgb.train(sampleData);
  const endTime: number = Date.now();
  
  const accuracy: number = xgb.evaluate(sampleData);
  console.log(`${alg.name}: ${accuracy.toFixed(3)} accuracy, ${endTime - startTime}ms`);
});

console.log('\n=== XGBoost TypeScript Feature Selection ===');

// Test different feature selection strategies with type safety
interface FeatureConfig {
  name: string;
  colsampleByTree: number;
}

const featureConfigs: FeatureConfig[] = [
  { name: 'All features', colsampleByTree: 1.0 },
  { name: '80% features', colsampleByTree: 0.8 },
  { name: '60% features', colsampleByTree: 0.6 },
  { name: '40% features', colsampleByTree: 0.4 }
];

featureConfigs.forEach((config: FeatureConfig) => {
  const xgb = new XGBoost(target, features, {
    nEstimators: 30,
    colsampleByTree: config.colsampleByTree,
    randomState: 42
  });
  xgb.train(sampleData);
  
  const accuracy: number = xgb.evaluate(sampleData);
  console.log(`${config.name}: ${accuracy.toFixed(3)} accuracy`);
});

console.log('\n=== XGBoost TypeScript Advanced Usage ===');

// Advanced usage with comprehensive configuration
const advancedConfig = {
  nEstimators: 100,
  learningRate: 0.05,
  maxDepth: 6,
  minChildWeight: 3,
  subsample: 0.9,
  colsampleByTree: 0.9,
  regAlpha: 0.2,
  regLambda: 2.0,
  objective: 'binary' as const,
  earlyStoppingRounds: 15,
  validationFraction: 0.25,
  randomState: 123
};

const xgbAdvanced = new XGBoost(target, features, advancedConfig);
xgbAdvanced.train(sampleData);

console.log('Advanced configuration training completed!');
console.log('Number of trees:', xgbAdvanced.getTreeCount());
console.log('Best iteration:', xgbAdvanced.getBestIteration());

// Get detailed feature importance
const advancedImportance = xgbAdvanced.getFeatureImportance();
console.log('Advanced feature importance:');
Object.entries(advancedImportance).forEach(([feature, importance]) => {
  console.log(`  ${feature}: ${importance.toFixed(4)}`);
});

// Get detailed boosting history
const advancedHistory = xgbAdvanced.getBoostingHistory();
console.log('Advanced boosting history:');
console.log('  Final training loss:', advancedHistory.trainLoss[advancedHistory.trainLoss.length - 1].toFixed(6));
console.log('  Final validation loss:', advancedHistory.validationLoss[advancedHistory.validationLoss.length - 1].toFixed(6));
console.log('  Total iterations:', advancedHistory.iterations.length);

console.log('\nXGBoost TypeScript examples completed successfully!');
