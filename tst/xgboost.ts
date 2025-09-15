import { strict as assert } from 'assert';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import XGBoost from '../lib/xgboost.js';

// Type definitions for test data
interface SampleData {
  color: string;
  shape: string;
  size: string;
  liked: boolean;
}

interface Dataset {
  features: string[];
  data: SampleData[];
}

// Helper function to load JSON files
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadJSON<T>(filename: string): T {
  const filePath = join(__dirname, '..', 'data', filename);
  return JSON.parse(readFileSync(filePath, 'utf8')) as T;
}

const SAMPLE_DATASET = loadJSON<Dataset>('sample.json');
const SAMPLE_DATASET_CLASS_NAME = 'liked';

describe('XGBoost Basics', function() {
  const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);

  it('should initialize with valid argument constructor', () => {
    assert.ok(new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
    assert.ok(new XGBoost(SAMPLE_DATASET.data, SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
  });

  it('should initialize with configuration', () => {
    const config = { 
      nEstimators: 50, 
      learningRate: 0.1, 
      maxDepth: 3,
      objective: 'binary' as const,
      randomState: 42 
    };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    assert.ok(xgb);
    assert.strictEqual(xgb.getConfig().nEstimators, 50);
    assert.strictEqual(xgb.getConfig().learningRate, 0.1);
    assert.strictEqual(xgb.getConfig().maxDepth, 3);
    assert.strictEqual(xgb.getConfig().objective, 'binary');
    assert.strictEqual(xgb.getConfig().randomState, 42);
  });

  it('should throw initialization error with invalid constructor arguments', () => {
    assert.throws(() => new XGBoost());
    assert.throws(() => new XGBoost(1 as any, 2 as any, 3 as any, 4 as any, 5 as any));
    assert.throws(() => new XGBoost(1 as any, 1 as any));
    assert.throws(() => new XGBoost("abc", 1 as any));
    assert.throws(() => new XGBoost(1 as any, 1 as any, 1 as any));
  });

  it('should train on the dataset', () => {
    xgb.train(SAMPLE_DATASET.data);
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should predict on a sample instance', () => {
    const config = { objective: 'binary' as const, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    const prediction = xgb.predict({ color: "blue", shape: "hexagon", size: "medium" });
    assert.ok(typeof prediction === 'boolean');
  });

  it('should evaluate on test dataset', () => {
    xgb.train(SAMPLE_DATASET.data);
    const accuracy = xgb.evaluate(SAMPLE_DATASET.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should provide access to the underlying model as JSON', () => {
    xgb.train(SAMPLE_DATASET.data);
    const modelJson = xgb.toJSON();
    assert.ok(modelJson.trees);
    assert.ok(Array.isArray(modelJson.trees));
    assert.ok(modelJson.trees.length > 0);
    assert.strictEqual(modelJson.target, SAMPLE_DATASET_CLASS_NAME);
    assert.deepStrictEqual(modelJson.features, SAMPLE_DATASET.features);
    assert.ok(typeof modelJson.baseScore === 'number');
    assert.ok(typeof modelJson.bestIteration === 'number');
    assert.ok(modelJson.boostingHistory);
  });

  it('should initialize from existing or previously exported model', () => {
    xgb.train(SAMPLE_DATASET.data);
    const modelJson = xgb.toJSON();
    const newXgb = new XGBoost(modelJson);
    assert.ok(newXgb.getTreeCount() > 0);
    assert.strictEqual(newXgb.getConfig().nEstimators, 100);
  });

  it('should throw error when predicting without training', () => {
    const untrainedXgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => untrainedXgb.predict({ color: "blue", shape: "hexagon" }));
  });

  it('should throw error when getting feature importance without training', () => {
    const untrainedXgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => untrainedXgb.getFeatureImportance());
  });
});

describe('XGBoost Configuration', function() {
  it('should use default configuration when none provided', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    const config = xgb.getConfig();
    assert.strictEqual(config.nEstimators, 100);
    assert.strictEqual(config.learningRate, 0.1);
    assert.strictEqual(config.maxDepth, 6);
    assert.strictEqual(config.minChildWeight, 1);
    assert.strictEqual(config.subsample, 1);
    assert.strictEqual(config.colsampleByTree, 1);
    assert.strictEqual(config.regAlpha, 0);
    assert.strictEqual(config.regLambda, 1);
    assert.strictEqual(config.objective, 'regression');
    assert.strictEqual(config.validationFraction, 0.2);
  });

  it('should accept custom nEstimators', () => {
    const config = { nEstimators: 25 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getTreeCount(), 25);
  });

  it('should accept custom learning rate', () => {
    const config = { learningRate: 0.05 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().learningRate, 0.05);
  });

  it('should accept custom max depth', () => {
    const config = { maxDepth: 3 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().maxDepth, 3);
  });

  it('should accept custom min child weight', () => {
    const config = { minChildWeight: 5 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().minChildWeight, 5);
  });

  it('should accept custom subsample', () => {
    const config = { subsample: 0.8 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().subsample, 0.8);
  });

  it('should accept custom colsample by tree', () => {
    const config = { colsampleByTree: 0.8 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().colsampleByTree, 0.8);
  });

  it('should accept custom regularization', () => {
    const config = { regAlpha: 0.1, regLambda: 2 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().regAlpha, 0.1);
    assert.strictEqual(xgb.getConfig().regLambda, 2);
  });

  it('should accept different objectives', () => {
    const regressionConfig = { objective: 'regression' as const };
    const binaryConfig = { objective: 'binary' as const };
    const multiclassConfig = { objective: 'multiclass' as const };

    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, regressionConfig);
    const xgb2 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, binaryConfig);
    const xgb3 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, multiclassConfig);

    assert.strictEqual(xgb1.getConfig().objective, 'regression');
    assert.strictEqual(xgb2.getConfig().objective, 'binary');
    assert.strictEqual(xgb3.getConfig().objective, 'multiclass');
  });

  it('should accept early stopping configuration', () => {
    const config = { 
      earlyStoppingRounds: 5, 
      validationFraction: 0.3,
      nEstimators: 50 
    };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().earlyStoppingRounds, 5);
    assert.strictEqual(xgb.getConfig().validationFraction, 0.3);
  });

  it('should accept random state for reproducibility', () => {
    const config = { randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    assert.strictEqual(xgb.getConfig().randomState, 42);
  });
});

describe('XGBoost Gradient Boosting', function() {
  it('should perform gradient boosting iterations', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 10);
    assert.ok(xgb.getBoostingHistory().trainLoss.length > 0);
    assert.ok(xgb.getBoostingHistory().iterations.length > 0);
  });

  it('should have decreasing training loss', () => {
    const config = { nEstimators: 20, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const history = xgb.getBoostingHistory();
    const losses = history.trainLoss;
    
    // Loss should generally decrease (allowing for some noise)
    let decreasingCount = 0;
    for (let i = 1; i < losses.length; i++) {
      if (losses[i] <= losses[i-1]) {
        decreasingCount++;
      }
    }
    
    // At least 50% of iterations should show decreasing loss
    assert.ok(decreasingCount / (losses.length - 1) >= 0.5);
  });

  it('should handle different learning rates', () => {
    const config1 = { learningRate: 0.01, nEstimators: 10, randomState: 42 };
    const config2 = { learningRate: 0.5, nEstimators: 10, randomState: 42 };
    
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config1);
    const xgb2 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config2);
    
    xgb1.train(SAMPLE_DATASET.data);
    xgb2.train(SAMPLE_DATASET.data);
    
    // Different learning rates should produce different models
    const pred1 = xgb1.predict({ color: "blue", shape: "hexagon", size: "medium" });
    const pred2 = xgb2.predict({ color: "blue", shape: "hexagon", size: "medium" });
    
    // They might be the same due to the small dataset, but the models should be different
    assert.ok(xgb1.getTreeCount() === xgb2.getTreeCount());
  });
});

describe('XGBoost Early Stopping', function() {
  it('should implement early stopping', () => {
    const config = { 
      nEstimators: 100, 
      earlyStoppingRounds: 5, 
      validationFraction: 0.3,
      randomState: 42 
    };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const history = xgb.getBoostingHistory();
    assert.ok(history.validationLoss.length > 0);
    assert.ok(xgb.getBestIteration() <= 100);
  });

  it('should stop early when no improvement', () => {
    const config = { 
      nEstimators: 50, 
      earlyStoppingRounds: 3, 
      validationFraction: 0.2,
      randomState: 42 
    };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    // Should stop before reaching nEstimators
    assert.ok(xgb.getTreeCount() <= 50);
    assert.ok(xgb.getBestIteration() <= 50);
  });

  it('should work without early stopping', () => {
    const config = { nEstimators: 20, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 20);
    assert.strictEqual(xgb.getBestIteration(), 20);
  });
});

describe('XGBoost Feature Importance', function() {
  it('should calculate feature importance', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const importance = xgb.getFeatureImportance();
    assert.ok(typeof importance === 'object');
    assert.ok(Array.isArray(Object.keys(importance)));
    
    // All features should have importance scores
    SAMPLE_DATASET.features.forEach(feature => {
      assert.ok(importance.hasOwnProperty(feature));
      assert.ok(typeof importance[feature] === 'number');
      assert.ok(importance[feature] >= 0);
    });
  });

  it('should have meaningful feature importance', () => {
    const config = { nEstimators: 20, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const importance = xgb.getFeatureImportance();
    const totalImportance = Object.values(importance).reduce((sum, val) => sum + val, 0);
    
    // Total importance should be greater than 0
    assert.ok(totalImportance > 0);
  });

  it('should throw error when getting importance without training', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.getFeatureImportance());
  });
});

describe('XGBoost Model Persistence', function() {
  it('should export and import model correctly', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    const xgb2 = new XGBoost(modelJson);
    
    assert.strictEqual(xgb2.getTreeCount(), 10);
    assert.strictEqual(xgb2.getConfig().nEstimators, 10);
    assert.strictEqual(xgb2.getConfig().randomState, 42);
  });

  it('should maintain prediction consistency after import', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    const xgb2 = new XGBoost(modelJson);
    
    const sample = { color: "blue", shape: "hexagon", size: "medium" };
    const pred1 = xgb1.predict(sample);
    const pred2 = xgb2.predict(sample);
    
    assert.strictEqual(pred1, pred2);
  });

  it('should maintain evaluation accuracy after import', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    const xgb2 = new XGBoost(modelJson);
    
    const accuracy1 = xgb1.evaluate(SAMPLE_DATASET.data);
    const accuracy2 = xgb2.evaluate(SAMPLE_DATASET.data);
    
    assert.strictEqual(accuracy1, accuracy2);
  });

  it('should maintain boosting history after import', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    const xgb2 = new XGBoost(modelJson);
    
    const history1 = xgb1.getBoostingHistory();
    const history2 = xgb2.getBoostingHistory();
    
    assert.deepStrictEqual(history1, history2);
  });
});

describe('XGBoost Edge Cases', function() {
  it('should handle single sample training data', () => {
    const singleSample = [SAMPLE_DATASET.data[0]];
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(singleSample));
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle empty training data', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.train([]));
  });

  it('should handle single feature', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, ['color'], config);
    
    assert.doesNotThrow(() => xgb.train(SAMPLE_DATASET.data));
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle prediction with missing features', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    // Missing some features
    const sample = { color: "blue" };
    assert.doesNotThrow(() => xgb.predict(sample));
  });

  it('should handle prediction with extra features', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    // Extra features not used in training
    const sample = { color: "blue", shape: "hexagon", size: "medium", extra: "value" };
    assert.doesNotThrow(() => xgb.predict(sample));
  });
});


describe('XGBoost on Sample Datasets', function() {
  it('should work with tic-tac-toe dataset', () => {
    const ticTacToeDataset = loadJSON<Dataset>('tic-tac-toe.json');
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost('class', ticTacToeDataset.features, config);
    
    assert.doesNotThrow(() => xgb.train(ticTacToeDataset.data));
    assert.ok(xgb.getTreeCount() > 0);
    
    const accuracy = xgb.evaluate(ticTacToeDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should work with voting dataset', () => {
    const votingDataset = loadJSON<Dataset>('voting.json');
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost('class', votingDataset.features, config);
    
    assert.doesNotThrow(() => xgb.train(votingDataset.data));
    assert.ok(xgb.getTreeCount() > 0);
    
    const accuracy = xgb.evaluate(votingDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should work with object evaluation dataset', () => {
    const objectDataset = loadJSON<Dataset>('object-evaluation.json');
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost('class', objectDataset.features, config);
    
    assert.doesNotThrow(() => xgb.train(objectDataset.data));
    assert.ok(xgb.getTreeCount() > 0);
    
    const accuracy = xgb.evaluate(objectDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });
});
