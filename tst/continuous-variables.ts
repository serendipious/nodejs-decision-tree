/**
 * Tests for Continuous Variable Support
 * Tests data type detection, CART algorithm, and hybrid functionality
 */

import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';
import { DataTypeDetector, detectDataTypes, recommendAlgorithm } from '../lib/shared/data-type-detection.js';
import { createCARTTree } from '../lib/shared/cart-algorithm.js';
import { globalCache } from '../lib/shared/caching-system.js';
import { processDataOptimized } from '../lib/shared/memory-optimization.js';

// Test data generators
function generateContinuousData(sampleCount: number): any[] {
  const data: any[] = [];
  for (let i = 0; i < sampleCount; i++) {
    data.push({
      age: Math.floor(Math.random() * 80) + 18,
      income: Math.random() * 100000 + 20000,
      score: Math.random() * 100,
      target: Math.random() > 0.5
    });
  }
  return data;
}

function generateDiscreteData(sampleCount: number): any[] {
  const colors = ['red', 'blue', 'green', 'yellow'];
  const shapes = ['circle', 'square', 'triangle'];
  const data: any[] = [];
  
  for (let i = 0; i < sampleCount; i++) {
    data.push({
      color: colors[Math.floor(Math.random() * colors.length)],
      shape: shapes[Math.floor(Math.random() * shapes.length)],
      target: Math.random() > 0.5
    });
  }
  return data;
}

function generateMixedData(sampleCount: number): any[] {
  const colors = ['red', 'blue', 'green'];
  const data: any[] = [];
  
  for (let i = 0; i < sampleCount; i++) {
    data.push({
      age: Math.floor(Math.random() * 80) + 18,
      income: Math.random() * 100000 + 20000,
      color: colors[Math.floor(Math.random() * colors.length)],
      target: Math.random() > 0.5
    });
  }
  return data;
}

function generateRegressionData(sampleCount: number): any[] {
  const data: any[] = [];
  for (let i = 0; i < sampleCount; i++) {
    const x1 = Math.random() * 10;
    const x2 = Math.random() * 5;
    const noise = (Math.random() - 0.5) * 0.5;
    const y = 2 * x1 + 3 * x2 + noise;
    
    data.push({
      x1,
      x2,
      target: y
    });
  }
  return data;
}

describe('Data Type Detection', function() {
  describe('Continuous Data Detection', function() {
    it('should detect continuous features correctly', function() {
      const data = generateContinuousData(100);
      const features = ['age', 'income', 'score'];
      
      const analysis = detectDataTypes(data, features);
      
      assert.strictEqual(analysis.age.type, 'continuous');
      assert.strictEqual(analysis.income.type, 'continuous');
      assert.strictEqual(analysis.score.type, 'continuous');
      assert(analysis.age.confidence > 0.7);
      assert(analysis.income.confidence > 0.7);
      assert(analysis.score.confidence > 0.7);
    });

    it('should detect discrete features correctly', function() {
      const data = generateDiscreteData(100);
      const features = ['color', 'shape'];
      
      const analysis = detectDataTypes(data, features);
      
      assert.strictEqual(analysis.color.type, 'discrete');
      assert.strictEqual(analysis.shape.type, 'discrete');
      assert(analysis.color.confidence > 0.7);
      assert(analysis.shape.confidence > 0.7);
    });

    it('should detect mixed data types correctly', function() {
      const data = generateMixedData(100);
      const features = ['age', 'income', 'color'];
      
      const analysis = detectDataTypes(data, features);
      
      assert.strictEqual(analysis.age.type, 'continuous');
      assert.strictEqual(analysis.income.type, 'continuous');
      assert.strictEqual(analysis.color.type, 'discrete');
    });

    it('should recommend appropriate algorithms', function() {
      const continuousData = generateContinuousData(100);
      const discreteData = generateDiscreteData(100);
      const mixedData = generateMixedData(100);
      
      const continuousRec = recommendAlgorithm(continuousData, ['age', 'income'], 'target');
      const discreteRec = recommendAlgorithm(discreteData, ['color', 'shape'], 'target');
      const mixedRec = recommendAlgorithm(mixedData, ['age', 'income', 'color'], 'target');
      
      assert.strictEqual(continuousRec.algorithm, 'cart');
      assert.strictEqual(discreteRec.algorithm, 'id3');
      assert.strictEqual(mixedRec.algorithm, 'hybrid');
    });
  });

  describe('Edge Cases', function() {
    it('should handle empty datasets', function() {
      const analysis = detectDataTypes([], ['feature1']);
      assert.strictEqual(analysis.feature1.type, 'discrete');
      assert.strictEqual(analysis.feature1.confidence, 0);
    });

    it('should handle single value datasets', function() {
      const data = [{ feature1: 5, target: true }];
      const analysis = detectDataTypes(data, ['feature1']);
      assert.strictEqual(analysis.feature1.type, 'discrete');
    });

    it('should handle missing values', function() {
      const data = [
        { feature1: 1, target: true },
        { feature1: null, target: false },
        { feature1: 3, target: true }
      ];
      const analysis = detectDataTypes(data, ['feature1']);
      assert(analysis.feature1.type === 'discrete' || analysis.feature1.type === 'continuous');
    });
  });
});

describe('Decision Tree with Continuous Variables', function() {
  describe('CART Algorithm', function() {
    it('should train on continuous data using CART', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      assert.strictEqual(dt.getAlgorithm(), 'cart');
      const featureTypes = dt.getFeatureTypes();
      assert.strictEqual(featureTypes.age, 'continuous');
      assert.strictEqual(featureTypes.income, 'continuous');
      assert.strictEqual(featureTypes.score, 'continuous');
    });

    it('should make predictions on continuous data', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const testSample = { age: 30, income: 50000, score: 75 };
      const prediction = dt.predict(testSample);
      
      assert(typeof prediction === 'boolean');
    });

    it('should handle regression tasks', function() {
      const data = generateRegressionData(100);
      const dt = new DecisionTree('target', ['x1', 'x2'], {
        algorithm: 'cart',
        criterion: 'mse',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const testSample = { x1: 5, x2: 2 };
      const prediction = dt.predict(testSample);
      
      assert(typeof prediction === 'number');
    });
  });

  describe('Hybrid Algorithm', function() {
    it('should automatically select hybrid approach for mixed data', function() {
      const data = generateMixedData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'color'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const featureTypes = dt.getFeatureTypes();
      assert.strictEqual(featureTypes.age, 'continuous');
      assert.strictEqual(featureTypes.income, 'continuous');
      assert.strictEqual(featureTypes.color, 'discrete');
    });

    it('should make predictions on mixed data', function() {
      const data = generateMixedData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'color'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const testSample = { age: 30, income: 50000, color: 'red' };
      const prediction = dt.predict(testSample);
      
      assert(typeof prediction === 'boolean');
    });
  });

  describe('Performance Tests', function() {
    it('should train on 1000 samples in < 50ms', function() {
      const data = generateContinuousData(1000);
      const start = Date.now();
      
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = Date.now() - start;
      assert(duration < 50, `Training took ${duration}ms, expected < 50ms`);
    });

    it('should predict on 100 samples in < 10ms', function() {
      const data = generateContinuousData(1000);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const testSamples = generateContinuousData(100);
      const start = Date.now();
      
      testSamples.forEach(sample => dt.predict(sample));
      
      const duration = Date.now() - start;
      assert(duration < 10, `Prediction took ${duration}ms, expected < 10ms`);
    });
  });
});

describe('Caching System', function() {
  beforeEach(function() {
    globalCache.clear();
  });

  describe('Prediction Caching', function() {
    it('should cache predictions for repeated samples', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      
      dt.train(data);
      
      const testSample = { age: 30, income: 50000, score: 75 };
      
      // First prediction (cold cache)
      const start1 = Date.now();
      const prediction1 = dt.predict(testSample);
      const duration1 = Date.now() - start1;
      
      // Second prediction (warm cache)
      const start2 = Date.now();
      const prediction2 = dt.predict(testSample);
      const duration2 = Date.now() - start2;
      
      assert.strictEqual(prediction1, prediction2);
      assert(duration2 < duration1, 'Cached prediction should be faster');
    });

    it('should provide cache statistics', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      
      dt.train(data);
      
      // Make some predictions
      const testSamples = generateContinuousData(10);
      testSamples.forEach(sample => dt.predict(sample));
      
      const stats = dt.getCacheStats();
      assert(stats !== null);
      assert(typeof stats.predictionCache.size === 'number');
    });
  });
});

describe('Memory Optimization', function() {
  describe('Optimized Dataset Processing', function() {
    it('should process continuous data efficiently', function() {
      const data = generateContinuousData(1000);
      const features = ['age', 'income', 'score'];
      const target = 'target';
      const featureTypes = new Map([
        ['age', 'continuous'],
        ['income', 'continuous'],
        ['score', 'continuous']
      ]);
      
      const optimizedDataset = processDataOptimized(data, features, target, featureTypes);
      
      assert.strictEqual(optimizedDataset.sampleCount, 1000);
      assert.strictEqual(optimizedDataset.featureCount, 3);
      assert(optimizedDataset.continuousFeatures.has('age'));
      assert(optimizedDataset.continuousFeatures.has('income'));
      assert(optimizedDataset.continuousFeatures.has('score'));
    });

    it('should process mixed data efficiently', function() {
      const data = generateMixedData(1000);
      const features = ['age', 'income', 'color'];
      const target = 'target';
      const featureTypes = new Map([
        ['age', 'continuous'],
        ['income', 'continuous'],
        ['color', 'discrete']
      ]);
      
      const optimizedDataset = processDataOptimized(data, features, target, featureTypes);
      
      assert.strictEqual(optimizedDataset.sampleCount, 1000);
      assert.strictEqual(optimizedDataset.featureCount, 3);
      assert(optimizedDataset.continuousFeatures.has('age'));
      assert(optimizedDataset.continuousFeatures.has('income'));
      assert(optimizedDataset.discreteFeatures.has('color'));
    });
  });
});

describe('Model Persistence', function() {
  describe('Continuous Variable Support', function() {
    it('should save and load models with continuous variables', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const modelJson = dt.toJSON();
      assert(modelJson.featureTypes !== undefined);
      assert(modelJson.algorithm !== undefined);
      assert(modelJson.config !== undefined);
      
      const loadedDt = new DecisionTree(modelJson);
      assert.strictEqual(loadedDt.getAlgorithm(), 'cart');
      
      const featureTypes = loadedDt.getFeatureTypes();
      assert.strictEqual(featureTypes.age, 'continuous');
      assert.strictEqual(featureTypes.income, 'continuous');
      assert.strictEqual(featureTypes.score, 'continuous');
    });

    it('should maintain prediction consistency after loading', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const testSample = { age: 30, income: 50000, score: 75 };
      const originalPrediction = dt.predict(testSample);
      
      const modelJson = dt.toJSON();
      const loadedDt = new DecisionTree(modelJson);
      const loadedPrediction = loadedDt.predict(testSample);
      
      assert.strictEqual(originalPrediction, loadedPrediction);
    });
  });
});

describe('Edge Cases and Error Handling', function() {
  describe('Invalid Data Handling', function() {
    it('should handle non-numeric continuous values gracefully', function() {
      const data = [
        { age: 25, income: 'invalid', target: true },
        { age: 30, income: 50000, target: false },
        { age: 35, income: 75000, target: true }
      ];
      
      const dt = new DecisionTree('target', ['age', 'income'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      // Should not throw an error
      assert.doesNotThrow(() => dt.train(data));
    });

    it('should handle missing features in prediction', function() {
      const data = generateContinuousData(100);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      
      dt.train(data);
      
      const testSample = { age: 30 }; // Missing income and score
      
      // Should not throw an error, should use fallback
      assert.doesNotThrow(() => dt.predict(testSample));
    });
  });

  describe('Configuration Validation', function() {
    it('should validate algorithm configuration', function() {
      const data = generateContinuousData(100);
      
      assert.throws(() => {
        new DecisionTree('target', ['age', 'income'], {
          algorithm: 'invalid' as any
        });
      });
    });

    it('should validate criterion configuration', function() {
      const data = generateContinuousData(100);
      
      assert.throws(() => {
        new DecisionTree('target', ['age', 'income'], {
          algorithm: 'cart',
          criterion: 'invalid' as any
        });
      });
    });
  });
});

describe('Performance Benchmarks', function() {
  describe('Training Performance', function() {
    it('should train on 10K samples in < 500ms', function() {
      const data = generateContinuousData(10000);
      const start = Date.now();
      
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = Date.now() - start;
      assert(duration < 500, `Training took ${duration}ms, expected < 500ms`);
    });

    it('should train on mixed data efficiently', function() {
      const data = generateMixedData(5000);
      const start = Date.now();
      
      const dt = new DecisionTree('target', ['age', 'income', 'color'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = Date.now() - start;
      assert(duration < 300, `Training took ${duration}ms, expected < 300ms`);
    });
  });

  describe('Inference Performance', function() {
    it('should predict 1000 samples in < 50ms', function() {
      const data = generateContinuousData(1000);
      const dt = new DecisionTree('target', ['age', 'income', 'score'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const testSamples = generateContinuousData(1000);
      const start = Date.now();
      
      testSamples.forEach(sample => dt.predict(sample));
      
      const duration = Date.now() - start;
      assert(duration < 50, `Prediction took ${duration}ms, expected < 50ms`);
    });
  });
});
