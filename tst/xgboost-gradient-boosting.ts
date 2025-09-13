import { strict as assert } from 'assert';
import { 
  createWeightedTree, 
  createWeightedSample, 
  calculateBaseScore 
} from '../lib/shared/gradient-boosting.js';
import { SeededRandom } from '../lib/shared/utils.js';

// Sample data for testing
const SAMPLE_DATA = [
  { color: 'red', shape: 'circle', size: 'small', target: 1 },
  { color: 'blue', shape: 'square', size: 'medium', target: 0 },
  { color: 'green', shape: 'triangle', size: 'large', target: 1 },
  { color: 'red', shape: 'square', size: 'small', target: 0 },
  { color: 'blue', shape: 'circle', size: 'medium', target: 1 }
];

const CONFIG = {
  nEstimators: 10,
  learningRate: 0.1,
  maxDepth: 3,
  minChildWeight: 1,
  subsample: 1,
  colsampleByTree: 1,
  regAlpha: 0,
  regLambda: 1,
  objective: 'regression' as const,
  randomState: 42
};

describe('XGBoost Gradient Boosting - createWeightedSample', function() {
  it('should create weighted sample with full data', () => {
    const random = new SeededRandom(42);
    const weightedSample = createWeightedSample(SAMPLE_DATA, CONFIG, random);
    
    assert.ok(Array.isArray(weightedSample.data));
    assert.ok(Array.isArray(weightedSample.weights));
    assert.ok(Array.isArray(weightedSample.gradients));
    assert.ok(Array.isArray(weightedSample.hessians));
    
    assert.strictEqual(weightedSample.data.length, SAMPLE_DATA.length);
    assert.strictEqual(weightedSample.weights.length, SAMPLE_DATA.length);
    assert.strictEqual(weightedSample.gradients.length, SAMPLE_DATA.length);
    assert.strictEqual(weightedSample.hessians.length, SAMPLE_DATA.length);
  });

  it('should create weighted sample with subsampling', () => {
    const config = { ...CONFIG, subsample: 0.8 };
    const random = new SeededRandom(42);
    const weightedSample = createWeightedSample(SAMPLE_DATA, config, random);
    
    assert.ok(weightedSample.data.length <= SAMPLE_DATA.length);
    assert.ok(weightedSample.data.length > 0);
  });

  it('should create weighted sample with feature sampling', () => {
    const config = { ...CONFIG, colsampleByTree: 0.5 };
    const random = new SeededRandom(42);
    const weightedSample = createWeightedSample(SAMPLE_DATA, config, random);
    
    assert.ok(Array.isArray(weightedSample.data));
    assert.ok(weightedSample.data.length > 0);
  });

  it('should handle empty data', () => {
    const random = new SeededRandom(42);
    const weightedSample = createWeightedSample([], CONFIG, random);
    
    assert.strictEqual(weightedSample.data.length, 0);
    assert.strictEqual(weightedSample.weights.length, 0);
    assert.strictEqual(weightedSample.gradients.length, 0);
    assert.strictEqual(weightedSample.hessians.length, 0);
  });

  it('should handle single sample', () => {
    const singleSample = [SAMPLE_DATA[0]];
    const random = new SeededRandom(42);
    const weightedSample = createWeightedSample(singleSample, CONFIG, random);
    
    assert.strictEqual(weightedSample.data.length, 1);
    assert.strictEqual(weightedSample.weights.length, 1);
    assert.strictEqual(weightedSample.gradients.length, 1);
    assert.strictEqual(weightedSample.hessians.length, 1);
  });

  it('should use different random seeds for different samples', () => {
    const config = { ...CONFIG, subsample: 0.8 }; // Enable subsampling to see differences
    const random1 = new SeededRandom(42);
    const random2 = new SeededRandom(123);
    
    const sample1 = createWeightedSample(SAMPLE_DATA, config, random1);
    const sample2 = createWeightedSample(SAMPLE_DATA, config, random2);
    
    // With different seeds, samples should be different
    assert.ok(JSON.stringify(sample1.data) !== JSON.stringify(sample2.data));
  });

  it('should use same random seed for reproducible samples', () => {
    const random1 = new SeededRandom(42);
    const random2 = new SeededRandom(42);
    
    const sample1 = createWeightedSample(SAMPLE_DATA, CONFIG, random1);
    const sample2 = createWeightedSample(SAMPLE_DATA, CONFIG, random2);
    
    // With same seed, samples should be identical
    assert.deepStrictEqual(sample1.data, sample2.data);
  });
});

describe('XGBoost Gradient Boosting - calculateBaseScore', function() {
  it('should calculate base score for regression', () => {
    const baseScore = calculateBaseScore(SAMPLE_DATA, 'target', 'regression');
    const expected = SAMPLE_DATA.reduce((sum, item) => sum + item.target, 0) / SAMPLE_DATA.length;
    assert.strictEqual(baseScore, expected);
  });

  it('should calculate base score for binary classification', () => {
    const binaryData = SAMPLE_DATA.map(item => ({ ...item, target: item.target === 1 ? 1 : 0 }));
    const baseScore = calculateBaseScore(binaryData, 'target', 'binary');
    
    // Should be log-odds of positive class probability
    const positiveCount = binaryData.filter(item => item.target === 1).length;
    const probability = positiveCount / binaryData.length;
    const expected = Math.log(probability / (1 - probability + 1e-15));
    
    assert.ok(Math.abs(baseScore - expected) < 1e-10);
  });

  it('should calculate base score for multiclass classification', () => {
    const baseScore = calculateBaseScore(SAMPLE_DATA, 'target', 'multiclass');
    assert.strictEqual(baseScore, 0);
  });

  it('should handle empty data', () => {
    const baseScore = calculateBaseScore([], 'target', 'regression');
    assert.ok(isNaN(baseScore));
  });

  it('should handle single sample', () => {
    const singleSample = [SAMPLE_DATA[0]];
    const baseScore = calculateBaseScore(singleSample, 'target', 'regression');
    assert.strictEqual(baseScore, singleSample[0].target);
  });

  it('should handle all positive samples for binary classification', () => {
    const allPositive = SAMPLE_DATA.map(item => ({ ...item, target: 1 }));
    const baseScore = calculateBaseScore(allPositive, 'target', 'binary');
    assert.ok(baseScore > 0);
  });

  it('should handle all negative samples for binary classification', () => {
    const allNegative = SAMPLE_DATA.map(item => ({ ...item, target: 0 }));
    const baseScore = calculateBaseScore(allNegative, 'target', 'binary');
    assert.ok(baseScore < 0);
  });

  it('should handle equal positive and negative samples for binary classification', () => {
    const equalData = [
      { color: 'red', target: 1 },
      { color: 'blue', target: 0 }
    ];
    const baseScore = calculateBaseScore(equalData, 'target', 'binary');
    assert.ok(Math.abs(baseScore) < 1e-10);
  });
});

describe('XGBoost Gradient Boosting - createWeightedTree', function() {
  it('should create weighted tree with valid data', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
    assert.ok(tree.type);
  });

  it('should create weighted tree with different weights', () => {
    const weights = [2, 1, 3, 1, 2];
    const gradients = [0.1, -0.2, 0.3, -0.1, 0.2];
    const hessians = [1, 1, 1, 1, 1];
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle empty data', () => {
    const weights: number[] = [];
    const gradients: number[] = [];
    const hessians: number[] = [];
    
    const tree = createWeightedTree(
      [],
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle single sample', () => {
    const singleSample = [SAMPLE_DATA[0]];
    const weights = [1];
    const gradients = [0.1];
    const hessians = [1];
    
    const tree = createWeightedTree(
      singleSample,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should respect max depth configuration', () => {
    const config = { ...CONFIG, maxDepth: 1 };
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      config
    );
    
    assert.ok(tree);
    // Tree should be limited by max depth
    assert.ok(getTreeDepth(tree) <= 1);
  });

  it('should respect min child weight configuration', () => {
    const config = { ...CONFIG, minChildWeight: 10 }; // Very high threshold
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      config
    );
    
    assert.ok(tree);
    // Tree should be a leaf due to high min child weight
    assert.strictEqual(tree.type, 'result');
  });

  it('should handle different regularization parameters', () => {
    const config = { ...CONFIG, regLambda: 0.5, regAlpha: 0.1 };
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      config
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });
});

describe('XGBoost Gradient Boosting - Edge Cases', function() {
  it('should handle zero weights', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(0);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle negative gradients', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = [-0.1, 0.2, -0.3, 0.1, -0.2];
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle zero hessians', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(1);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(0);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle very large weights', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(1e6);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });

  it('should handle very small weights', () => {
    const weights = new Array(SAMPLE_DATA.length).fill(1e-6);
    const gradients = new Array(SAMPLE_DATA.length).fill(0.1);
    const hessians = new Array(SAMPLE_DATA.length).fill(1);
    
    const tree = createWeightedTree(
      SAMPLE_DATA,
      'target',
      ['color', 'shape', 'size'],
      weights,
      gradients,
      hessians,
      CONFIG
    );
    
    assert.ok(tree);
    assert.ok(typeof tree === 'object');
  });
});

// Helper function to calculate tree depth
function getTreeDepth(node: any): number {
  if (node.type === 'result') {
    return 0;
  }
  
  if (node.vals && node.vals.length > 0) {
    let maxChildDepth = 0;
    for (const val of node.vals) {
      if (val.child) {
        maxChildDepth = Math.max(maxChildDepth, getTreeDepth(val.child));
      }
    }
    return 1 + maxChildDepth;
  }
  
  return 0;
}
