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

describe('XGBoost Edge Cases - Empty and Invalid Datasets', function() {
  it('should handle empty training dataset', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.train([]));
  });

  it('should handle null training data', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.train(null as any));
  });

  it('should handle undefined training data', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.train(undefined as any));
  });

  it('should handle non-array training data', () => {
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
    assert.throws(() => xgb.train({} as any));
    assert.throws(() => xgb.train('string' as any));
    assert.throws(() => xgb.train(123 as any));
  });

  it('should handle single sample training data', () => {
    const singleSample = [SAMPLE_DATASET.data[0]];
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(singleSample));
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle training data with missing features', () => {
    const incompleteData = [
      { color: 'red', shape: 'circle' }, // Missing size
      { color: 'blue', shape: 'square', size: 'medium' }
    ];
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(incompleteData));
  });

  it('should handle training data with extra features', () => {
    const extraData = SAMPLE_DATASET.data.map(item => ({
      ...item,
      extraFeature: 'extra'
    }));
    const config = { nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(extraData));
  });
});

describe('XGBoost Edge Cases - Configuration Edge Cases', function() {
  it('should handle zero nEstimators', () => {
    const config = { nEstimators: 0, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 0);
    assert.strictEqual(xgb.getBestIteration(), 0);
  });

  it('should handle negative nEstimators', () => {
    const config = { nEstimators: -5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 0);
    assert.strictEqual(xgb.getBestIteration(), 0);
  });

  it('should handle very large nEstimators', () => {
    const config = { nEstimators: 1000, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    const startTime = Date.now();
    xgb.train(SAMPLE_DATASET.data);
    const endTime = Date.now();
    
    assert.strictEqual(xgb.getTreeCount(), 1000);
    assert.ok(endTime - startTime < 30000); // Should complete within 30 seconds
  });

  it('should handle zero learning rate', () => {
    const config = { learningRate: 0, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    // With zero learning rate, predictions should be base score
    const prediction = xgb.predict({ color: 'blue', shape: 'hexagon', size: 'medium' });
    assert.ok(typeof prediction === 'number');
  });

  it('should handle very small learning rate', () => {
    const config = { learningRate: 1e-10, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very large learning rate', () => {
    const config = { learningRate: 10, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero max depth', () => {
    const config = { maxDepth: 0, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very large max depth', () => {
    const config = { maxDepth: 100, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero min child weight', () => {
    const config = { minChildWeight: 0, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very large min child weight', () => {
    const config = { minChildWeight: 1000, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero subsample', () => {
    const config = { subsample: 0, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle subsample greater than 1', () => {
    const config = { subsample: 1.5, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero colsample by tree', () => {
    const config = { colsampleByTree: 0, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle colsample by tree greater than 1', () => {
    const config = { colsampleByTree: 1.5, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle negative regularization', () => {
    const config = { regAlpha: -0.1, regLambda: -0.1, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very large regularization', () => {
    const config = { regAlpha: 1000, regLambda: 1000, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero validation fraction', () => {
    const config = { validationFraction: 0, earlyStoppingRounds: 5, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle validation fraction greater than 1', () => {
    const config = { validationFraction: 1.5, earlyStoppingRounds: 5, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle zero early stopping rounds', () => {
    const config = { earlyStoppingRounds: 0, validationFraction: 0.2, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 10);
  });

  it('should handle early stopping rounds greater than nEstimators', () => {
    const config = { earlyStoppingRounds: 50, validationFraction: 0.2, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.strictEqual(xgb.getTreeCount(), 10);
  });
});

describe('XGBoost Edge Cases - Prediction Edge Cases', function() {
  let xgb: XGBoost;

  beforeEach(() => {
    const config = { nEstimators: 10, randomState: 42 };
    xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
  });

  it('should handle prediction with null sample', () => {
    assert.throws(() => xgb.predict(null as any));
  });

  it('should handle prediction with undefined sample', () => {
    assert.throws(() => xgb.predict(undefined as any));
  });

  it('should handle prediction with non-object sample', () => {
    assert.throws(() => xgb.predict('string' as any), /Sample must be an object/);
  });

  it('should handle prediction with numeric sample', () => {
    assert.throws(() => xgb.predict(123 as any), /Sample must be an object/);
  });

  it('should handle prediction with array sample', () => {
    assert.throws(() => xgb.predict([] as any), /Sample must be an object/);
  });

  it('should handle prediction with empty sample', () => {
    const prediction = xgb.predict({});
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });

  it('should handle prediction with missing all features', () => {
    const prediction = xgb.predict({ unrelated: 'value' });
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });

  it('should handle prediction with some missing features', () => {
    const prediction = xgb.predict({ color: 'blue' });
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });

  it('should handle prediction with extra features', () => {
    const prediction = xgb.predict({ 
      color: 'blue', 
      shape: 'hexagon', 
      size: 'medium',
      extra: 'value',
      another: 123
    });
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });

  it('should handle prediction with null feature values', () => {
    const prediction = xgb.predict({ 
      color: null, 
      shape: 'hexagon', 
      size: 'medium'
    });
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });

  it('should handle prediction with undefined feature values', () => {
    const prediction = xgb.predict({ 
      color: undefined, 
      shape: 'hexagon', 
      size: 'medium'
    });
    assert.ok(typeof prediction === 'boolean' || typeof prediction === 'number');
  });
});

describe('XGBoost Edge Cases - Model Persistence Edge Cases', function() {
  it('should handle importing model with missing properties', () => {
    const invalidModel = {
      trees: [],
      target: 'liked',
      features: ['color', 'shape', 'size'],
      config: {},
      data: []
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with invalid trees', () => {
    const invalidModel = {
      trees: 'invalid',
      target: 'liked',
      features: ['color', 'shape', 'size'],
      config: {},
      data: [],
      baseScore: 0,
      bestIteration: 0,
      boostingHistory: { trainLoss: [], validationLoss: [], iterations: [] }
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with invalid target', () => {
    const invalidModel = {
      trees: [],
      target: 123,
      features: ['color', 'shape', 'size'],
      config: {},
      data: [],
      baseScore: 0,
      bestIteration: 0,
      boostingHistory: { trainLoss: [], validationLoss: [], iterations: [] }
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with invalid features', () => {
    const invalidModel = {
      trees: [],
      target: 'liked',
      features: 'invalid',
      config: {},
      data: [],
      baseScore: 0,
      bestIteration: 0,
      boostingHistory: { trainLoss: [], validationLoss: [], iterations: [] }
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with invalid config', () => {
    const invalidModel = {
      trees: [],
      target: 'liked',
      features: ['color', 'shape', 'size'],
      config: 'invalid',
      data: [],
      baseScore: 0,
      bestIteration: 0,
      boostingHistory: { trainLoss: [], validationLoss: [], iterations: [] }
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with invalid data', () => {
    const invalidModel = {
      trees: [],
      target: 'liked',
      features: ['color', 'shape', 'size'],
      config: {},
      data: 'invalid',
      baseScore: 0,
      bestIteration: 0,
      boostingHistory: { trainLoss: [], validationLoss: [], iterations: [] }
    };
    
    assert.throws(() => new XGBoost(invalidModel as any));
  });

  it('should handle importing model with corrupted tree data', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    // Corrupt one of the trees
    modelJson.trees[0].model = null as any;
    
    const xgb2 = new XGBoost(modelJson);
    // Should not throw during import, but predict should throw
    assert.throws(() => xgb2.predict({ color: 'blue', shape: 'hexagon', size: 'medium' }));
  });

  it('should handle importing model with missing baseScore', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    delete (modelJson as any).baseScore;
    
    assert.throws(() => new XGBoost(modelJson), /baseScore property is required/);
  });

  it('should handle importing model with missing bestIteration', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    delete (modelJson as any).bestIteration;
    
    assert.throws(() => new XGBoost(modelJson), /bestIteration property is required/);
  });

  it('should handle importing model with missing boostingHistory', () => {
    const config = { nEstimators: 5, randomState: 42 };
    const xgb1 = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb1.train(SAMPLE_DATASET.data);
    
    const modelJson = xgb1.toJSON();
    delete (modelJson as any).boostingHistory;
    
    assert.throws(() => new XGBoost(modelJson), /boostingHistory property is required/);
  });
});

describe('XGBoost Edge Cases - Feature Importance Edge Cases', function() {
  it('should handle feature importance with zero trees', () => {
    const config = { nEstimators: 0, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    assert.throws(() => xgb.getFeatureImportance());
  });

  it('should handle feature importance with single tree', () => {
    const config = { nEstimators: 1, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const importance = xgb.getFeatureImportance();
    assert.ok(typeof importance === 'object');
    assert.ok(Array.isArray(Object.keys(importance)));
  });

  it('should handle feature importance with many trees', () => {
    const config = { nEstimators: 100, randomState: 42 };
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
});

describe('XGBoost Edge Cases - Boosting History Edge Cases', function() {
  it('should handle boosting history with zero trees', () => {
    const config = { nEstimators: 0, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const history = xgb.getBoostingHistory();
    assert.ok(Array.isArray(history.trainLoss));
    assert.ok(Array.isArray(history.validationLoss));
    assert.ok(Array.isArray(history.iterations));
    assert.strictEqual(history.trainLoss.length, 0);
    assert.strictEqual(history.validationLoss.length, 0);
    assert.strictEqual(history.iterations.length, 0);
  });

  it('should handle boosting history with single tree', () => {
    const config = { nEstimators: 1, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const history = xgb.getBoostingHistory();
    assert.strictEqual(history.trainLoss.length, 1);
    assert.strictEqual(history.iterations.length, 1);
    assert.ok(history.trainLoss[0] >= 0);
    assert.strictEqual(history.iterations[0], 1);
  });

  it('should handle boosting history with early stopping', () => {
    const config = { 
      nEstimators: 50, 
      earlyStoppingRounds: 3, 
      validationFraction: 0.2,
      randomState: 42 
    };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    xgb.train(SAMPLE_DATASET.data);
    
    const history = xgb.getBoostingHistory();
    assert.ok(history.trainLoss.length > 0);
    assert.ok(history.iterations.length > 0);
    assert.ok(history.iterations.length <= 50);
  });
});

describe('XGBoost Edge Cases - Performance Edge Cases', function() {
  it('should handle very large datasets', () => {
    // Create a large dataset
    const largeData = [];
    for (let i = 0; i < 1000; i++) {
      largeData.push({
        color: ['red', 'blue', 'green'][i % 3],
        shape: ['circle', 'square', 'triangle'][i % 3],
        size: ['small', 'medium', 'large'][i % 3],
        liked: i % 2 === 0
      });
    }
    
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost('liked', ['color', 'shape', 'size'], config);
    
    const startTime = Date.now();
    xgb.train(largeData);
    const endTime = Date.now();
    
    assert.ok(xgb.getTreeCount() > 0);
    assert.ok(endTime - startTime < 10000); // Should complete within 10 seconds
  });

  it('should handle many features', () => {
    const manyFeatures = [
      'color', 'shape', 'size', 'texture', 'weight', 'height', 'width', 'density',
      'temperature', 'humidity', 'pressure', 'speed', 'direction', 'intensity'
    ];
    
    const config = { nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost('liked', manyFeatures, config);
    
    // Create data with many features
    const dataWithManyFeatures = SAMPLE_DATASET.data.map(item => {
      const newItem: any = { ...item };
      manyFeatures.forEach(feature => {
        if (!newItem[feature]) {
          newItem[feature] = `value_${Math.random()}`;
        }
      });
      return newItem;
    });
    
    assert.doesNotThrow(() => xgb.train(dataWithManyFeatures));
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very deep trees', () => {
    const config = { maxDepth: 20, nEstimators: 5, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(SAMPLE_DATASET.data));
    assert.ok(xgb.getTreeCount() > 0);
  });

  it('should handle very shallow trees', () => {
    const config = { maxDepth: 1, nEstimators: 10, randomState: 42 };
    const xgb = new XGBoost(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features, config);
    
    assert.doesNotThrow(() => xgb.train(SAMPLE_DATASET.data));
    assert.ok(xgb.getTreeCount() > 0);
  });
});
