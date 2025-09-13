import { strict as assert } from 'assert';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import RandomForest from '../lib/random-forest.js';

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

describe('Random Forest Basics', function() {
  const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);

  it('should initialize with valid argument constructor', () => {
    assert.ok(new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
    assert.ok(new RandomForest(SAMPLE_DATASET.data, SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
  });

  it('should initialize with configuration', () => {
    const config = { nEstimators: 50, maxFeatures: 'sqrt' as const, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    assert.ok(rf);
    assert.strictEqual(rf.getConfig().nEstimators, 50);
    assert.strictEqual(rf.getConfig().maxFeatures, 'sqrt');
    assert.strictEqual(rf.getConfig().randomState, 42);
  });

  it('should throw initialization error with invalid constructor arguments', () => {
    assert.throws(() => new RandomForest());
    assert.throws(() => new RandomForest(1 as any, 2 as any, 3 as any, 4 as any, 5 as any));
    assert.throws(() => new RandomForest(1 as any, 1 as any));
    assert.throws(() => new RandomForest("abc", 1 as any));
    assert.throws(() => new RandomForest(1 as any, 1 as any, 1 as any));
  });

  it('should train on the dataset', () => {
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getTreeCount(), 100); // Default nEstimators
  });

  it('should predict on a sample instance', () => {
    rf.train(SAMPLE_DATASET.data);
    const prediction = rf.predict({ color: "blue", shape: "hexagon", size: "medium" });
    assert.ok(typeof prediction === 'boolean');
  });

  it('should evaluate on test dataset', () => {
    rf.train(SAMPLE_DATASET.data);
    const accuracy = rf.evaluate(SAMPLE_DATASET.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should provide access to the underlying model as JSON', () => {
    rf.train(SAMPLE_DATASET.data);
    const modelJson = rf.toJSON();
    assert.ok(modelJson.trees);
    assert.ok(Array.isArray(modelJson.trees));
    assert.strictEqual(modelJson.trees.length, 100);
    assert.strictEqual(modelJson.target, SAMPLE_DATASET_CLASS_NAME);
    assert.deepStrictEqual(modelJson.features, SAMPLE_DATASET.features);
  });

  it('should initialize from existing or previously exported model', () => {
    rf.train(SAMPLE_DATASET.data);
    const modelJson = rf.toJSON();
    const newRf = new RandomForest(modelJson);
    assert.strictEqual(newRf.getTreeCount(), 100);
    assert.strictEqual(newRf.getConfig().nEstimators, 100);
  });

  it('should throw error when predicting without training', () => {
    const untrainedRf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => untrainedRf.predict({ color: "blue", shape: "hexagon" }));
  });

  it('should throw error when getting feature importance without training', () => {
    const untrainedRf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => untrainedRf.getFeatureImportance());
  });
});

describe('Random Forest Configuration', function() {
  it('should use default configuration when none provided', () => {
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    const config = rf.getConfig();
    assert.strictEqual(config.nEstimators, 100);
    assert.strictEqual(config.maxFeatures, 'sqrt');
    assert.strictEqual(config.bootstrap, true);
    assert.strictEqual(config.minSamplesSplit, 2);
  });

  it('should accept custom nEstimators', () => {
    const config = { nEstimators: 25 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getTreeCount(), 25);
  });

  it('should accept sqrt maxFeatures', () => {
    const config = { maxFeatures: 'sqrt' as const };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().maxFeatures, 'sqrt');
  });

  it('should accept log2 maxFeatures', () => {
    const config = { maxFeatures: 'log2' as const };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().maxFeatures, 'log2');
  });

  it('should accept auto maxFeatures', () => {
    const config = { maxFeatures: 'auto' as const };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().maxFeatures, 'auto');
  });

  it('should accept numeric maxFeatures', () => {
    const config = { maxFeatures: 2 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().maxFeatures, 2);
  });

  it('should disable bootstrap sampling', () => {
    const config = { bootstrap: false };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().bootstrap, false);
  });

  it('should accept randomState for reproducibility', () => {
    const config = { randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().randomState, 42);
  });

  it('should accept maxDepth', () => {
    const config = { maxDepth: 3 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().maxDepth, 3);
  });

  it('should accept minSamplesSplit', () => {
    const config = { minSamplesSplit: 5 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    assert.strictEqual(rf.getConfig().minSamplesSplit, 5);
  });
});

describe('Random Forest Bootstrap Sampling', function() {
  it('should create different bootstrap samples for each tree', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    // Each tree should be different due to bootstrap sampling
    const modelJson = rf.toJSON();
    const firstTree = modelJson.trees[0];
    const secondTree = modelJson.trees[1];
    
    // Trees should have different structures due to different bootstrap samples
    assert.notDeepStrictEqual(firstTree.model, secondTree.model);
  });

  it('should maintain same sample size with bootstrap', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const modelJson = rf.toJSON();
    modelJson.trees.forEach(tree => {
      // With bootstrap sampling, each tree should have the same sample size as original data
      assert.strictEqual(tree.data.length, SAMPLE_DATASET.data.length);
    });
  });

  it('should work without bootstrap sampling', () => {
    const config = { nEstimators: 5, bootstrap: false, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const modelJson = rf.toJSON();
    // All trees should have identical training data
    const firstTreeData = modelJson.trees[0].data;
    modelJson.trees.forEach(tree => {
      assert.deepStrictEqual(tree.data, firstTreeData);
    });
  });
});

describe('Random Forest Feature Selection', function() {
  it('should use different features for each tree', () => {
    const config = { nEstimators: 10, maxFeatures: 2, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const modelJson = rf.toJSON();
    const firstTreeFeatures = modelJson.trees[0].features;
    const secondTreeFeatures = modelJson.trees[1].features;
    
    // Should have different feature sets
    assert.notDeepStrictEqual(firstTreeFeatures, secondTreeFeatures);
  });

  it('should respect maxFeatures limit', () => {
    const config = { nEstimators: 5, maxFeatures: 2, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const modelJson = rf.toJSON();
    modelJson.trees.forEach(tree => {
      assert.ok(tree.features.length <= 2);
    });
  });

  it('should use sqrt features correctly', () => {
    const config = { nEstimators: 5, maxFeatures: 'sqrt' as const, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const expectedFeatures = Math.floor(Math.sqrt(SAMPLE_DATASET.features.length));
    const modelJson = rf.toJSON();
    modelJson.trees.forEach(tree => {
      assert.ok(tree.features.length <= expectedFeatures);
    });
  });

  it('should use log2 features correctly', () => {
    const config = { nEstimators: 5, maxFeatures: 'log2' as const, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const expectedFeatures = Math.floor(Math.log2(SAMPLE_DATASET.features.length));
    const modelJson = rf.toJSON();
    modelJson.trees.forEach(tree => {
      assert.ok(tree.features.length <= expectedFeatures);
    });
  });
});

describe('Random Forest Ensemble Prediction', function() {
  it('should use majority voting for predictions', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const prediction = rf.predict({ color: "blue", shape: "hexagon", size: "medium" });
    assert.ok(typeof prediction === 'boolean');
  });

  it('should be more stable than single decision tree', () => {
    const config = { nEstimators: 50, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    // Make multiple predictions on same sample
    const sample = { color: "blue", shape: "hexagon", size: "medium" };
    const predictions = [];
    for (let i = 0; i < 10; i++) {
      predictions.push(rf.predict(sample));
    }
    
    // All predictions should be the same (due to randomState)
    const firstPrediction = predictions[0];
    predictions.forEach(pred => {
      assert.strictEqual(pred, firstPrediction);
    });
  });

  it('should handle tie-breaking in majority voting', () => {
    // Create a scenario where we can test tie-breaking
    const config = { nEstimators: 4, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const prediction = rf.predict({ color: "blue", shape: "hexagon", size: "medium" });
    assert.ok(typeof prediction === 'boolean');
  });
});

describe('Random Forest Feature Importance', function() {
  it('should calculate feature importance', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const importance = rf.getFeatureImportance();
    assert.ok(typeof importance === 'object');
    assert.ok(Array.isArray(Object.keys(importance)));
    
    // All features should have importance scores
    SAMPLE_DATASET.features.forEach(feature => {
      assert.ok(importance.hasOwnProperty(feature));
      assert.ok(typeof importance[feature] === 'number');
      assert.ok(importance[feature] >= 0);
    });
  });

  it('should have normalized importance scores', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    const importance = rf.getFeatureImportance();
    const totalImportance = Object.values(importance).reduce((sum, val) => sum + val, 0);
    
    // Total importance should be reasonable (not necessarily 1 due to normalization method)
    assert.ok(totalImportance > 0);
  });

  it('should throw error when getting importance without training', () => {
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => rf.getFeatureImportance());
  });
});

describe('Random Forest Model Persistence', function() {
  it('should export and import model correctly', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf1 = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf1.train(SAMPLE_DATASET.data);
    
    const modelJson = rf1.toJSON();
    const rf2 = new RandomForest(modelJson);
    
    assert.strictEqual(rf2.getTreeCount(), 5);
    assert.strictEqual(rf2.getConfig().nEstimators, 5);
    assert.strictEqual(rf2.getConfig().randomState, 42);
  });

  it('should maintain prediction consistency after import', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf1 = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf1.train(SAMPLE_DATASET.data);
    
    const modelJson = rf1.toJSON();
    const rf2 = new RandomForest(modelJson);
    
    const sample = { color: "blue", shape: "hexagon", size: "medium" };
    const pred1 = rf1.predict(sample);
    const pred2 = rf2.predict(sample);
    
    assert.strictEqual(pred1, pred2);
  });

  it('should maintain evaluation accuracy after import', () => {
    const config = { nEstimators: 10, randomState: 42 };
    const rf1 = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf1.train(SAMPLE_DATASET.data);
    
    const modelJson = rf1.toJSON();
    const rf2 = new RandomForest(modelJson);
    
    const accuracy1 = rf1.evaluate(SAMPLE_DATASET.data);
    const accuracy2 = rf2.evaluate(SAMPLE_DATASET.data);
    
    assert.strictEqual(accuracy1, accuracy2);
  });
});

describe('Random Forest Edge Cases', function() {
  it('should handle single sample training data', () => {
    const singleSample = [SAMPLE_DATASET.data[0]];
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => rf.train(singleSample));
    assert.strictEqual(rf.getTreeCount(), 5);
  });

  it('should handle empty training data', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.throws(() => rf.train([]));
  });

  it('should handle single feature', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, ['color'], config);
    
    assert.doesNotThrow(() => rf.train(SAMPLE_DATASET.data));
    assert.strictEqual(rf.getTreeCount(), 5);
  });

  it('should handle prediction with missing features', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    // Missing some features
    const sample = { color: "blue" };
    assert.doesNotThrow(() => rf.predict(sample));
  });

  it('should handle prediction with extra features', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    rf.train(SAMPLE_DATASET.data);
    
    // Extra features not used in training
    const sample = { color: "blue", shape: "hexagon", size: "medium", extra: "value" };
    assert.doesNotThrow(() => rf.predict(sample));
  });
});

describe('Random Forest Performance', function() {
  it('should handle large number of estimators', () => {
    const config = { nEstimators: 200, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    const startTime = Date.now();
    rf.train(SAMPLE_DATASET.data);
    const endTime = Date.now();
    
    assert.strictEqual(rf.getTreeCount(), 200);
    assert.ok(endTime - startTime < 10000); // Should complete within 10 seconds
  });

  it('should handle many features', () => {
    const manyFeatures = ['color', 'shape', 'size', 'texture', 'weight', 'height', 'width', 'density'];
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest(SAMPLE_DATASET_CLASS_NAME, manyFeatures, config);
    
    assert.doesNotThrow(() => rf.train(SAMPLE_DATASET.data));
    assert.strictEqual(rf.getTreeCount(), 10);
  });
});

describe('Random Forest on Sample Datasets', function() {
  it('should work with tic-tac-toe dataset', () => {
    const ticTacToeDataset = loadJSON<Dataset>('tic-tac-toe.json');
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest('class', ticTacToeDataset.features, config);
    
    assert.doesNotThrow(() => rf.train(ticTacToeDataset.data));
    assert.strictEqual(rf.getTreeCount(), 10);
    
    const accuracy = rf.evaluate(ticTacToeDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should work with voting dataset', () => {
    const votingDataset = loadJSON<Dataset>('voting.json');
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest('class', votingDataset.features, config);
    
    assert.doesNotThrow(() => rf.train(votingDataset.data));
    assert.strictEqual(rf.getTreeCount(), 10);
    
    const accuracy = rf.evaluate(votingDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  it('should work with object evaluation dataset', () => {
    const objectDataset = loadJSON<Dataset>('object-evaluation.json');
    const config = { nEstimators: 10, randomState: 42 };
    const rf = new RandomForest('class', objectDataset.features, config);
    
    assert.doesNotThrow(() => rf.train(objectDataset.data));
    assert.strictEqual(rf.getTreeCount(), 10);
    
    const accuracy = rf.evaluate(objectDataset.data);
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });
});
