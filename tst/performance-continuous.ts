/**
 * Performance Tests for Continuous Variables
 * Tests low-latency requirements, caching effectiveness, and memory optimization
 */

import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';
import { globalCache } from '../lib/shared/caching-system.js';
import { processDataOptimized } from '../lib/shared/memory-optimization.js';

// Performance test data generators
function generateLargeContinuousData(sampleCount: number): any[] {
  const data: any[] = [];
  for (let i = 0; i < sampleCount; i++) {
    data.push({
      feature1: Math.random() * 100,
      feature2: Math.random() * 50,
      feature3: Math.random() * 200,
      feature4: Math.random() * 75,
      feature5: Math.random() * 150,
      target: Math.random() > 0.5
    });
  }
  return data;
}

function generateLargeMixedData(sampleCount: number): any[] {
  const categories = ['A', 'B', 'C', 'D', 'E'];
  const data: any[] = [];
  
  for (let i = 0; i < sampleCount; i++) {
    data.push({
      continuous1: Math.random() * 100,
      continuous2: Math.random() * 50,
      discrete1: categories[Math.floor(Math.random() * categories.length)],
      discrete2: Math.random() > 0.5,
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
    const x3 = Math.random() * 8;
    const noise = (Math.random() - 0.5) * 0.5;
    const y = 2 * x1 + 3 * x2 - 1.5 * x3 + noise;
    
    data.push({
      x1,
      x2,
      x3,
      target: y
    });
  }
  return data;
}

describe('Training Latency Tests', function() {
  describe('Small Dataset Performance', function() {
    it('should train DecisionTree on 100 samples in < 10ms', function() {
      const data = generateLargeContinuousData(100);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 10, `Training took ${duration}ms, expected < 10ms`);
    });

    it('should train DecisionTree on 1000 samples in < 50ms', function() {
      const data = generateLargeContinuousData(1000);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 50, `Training took ${duration}ms, expected < 50ms`);
    });

    it('should train DecisionTree on mixed data in < 75ms', function() {
      const data = generateLargeMixedData(1000);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['continuous1', 'continuous2', 'discrete1', 'discrete2'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 75, `Training took ${duration}ms, expected < 75ms`);
    });
  });

  describe('Medium Dataset Performance', function() {
    it('should train DecisionTree on 10K samples in < 500ms', function() {
      const data = generateLargeContinuousData(10000);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 500, `Training took ${duration}ms, expected < 500ms`);
    });

    it('should train DecisionTree on regression data in < 300ms', function() {
      const data = generateRegressionData(5000);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['x1', 'x2', 'x3'], {
        algorithm: 'cart',
        criterion: 'mse',
        autoDetectTypes: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 300, `Training took ${duration}ms, expected < 300ms`);
    });
  });

  describe('Large Dataset Performance', function() {
    it('should train DecisionTree on 100K samples in < 5s', function() {
      const data = generateLargeContinuousData(100000);
      const start = performance.now();
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        memoryOptimization: true
      });
      dt.train(data);
      
      const duration = performance.now() - start;
      assert(duration < 5000, `Training took ${duration}ms, expected < 5s`);
    });
  });
});

describe('Inference Latency Tests', function() {
  let trainedModel: DecisionTree;

  beforeEach(function() {
    const data = generateLargeContinuousData(1000);
    trainedModel = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
      algorithm: 'cart',
      autoDetectTypes: true,
      cachingEnabled: true
    });
    trainedModel.train(data);
  });

  describe('Single Prediction Performance', function() {
    it('should predict on DecisionTree in < 1ms', function() {
      const sample = {
        feature1: Math.random() * 100,
        feature2: Math.random() * 50,
        feature3: Math.random() * 200
      };
      
      const start = performance.now();
      const prediction = trainedModel.predict(sample);
      const duration = performance.now() - start;
      
      assert(duration < 1, `Prediction took ${duration}ms, expected < 1ms`);
      assert(typeof prediction === 'boolean');
    });

    it('should predict on regression model in < 1ms', function() {
      const regressionData = generateRegressionData(1000);
      const regressionModel = new DecisionTree('target', ['x1', 'x2', 'x3'], {
        algorithm: 'cart',
        criterion: 'mse',
        autoDetectTypes: true
      });
      regressionModel.train(regressionData);
      
      const sample = { x1: 5, x2: 2, x3: 3 };
      const start = performance.now();
      const prediction = regressionModel.predict(sample);
      const duration = performance.now() - start;
      
      assert(duration < 1, `Prediction took ${duration}ms, expected < 1ms`);
      assert(typeof prediction === 'number');
    });
  });

  describe('Batch Prediction Performance', function() {
    it('should predict 100 samples in < 10ms', function() {
      const samples = generateLargeContinuousData(100);
      
      const start = performance.now();
      const predictions = samples.map(sample => trainedModel.predict(sample));
      const duration = performance.now() - start;
      
      assert(duration < 10, `Batch prediction took ${duration}ms, expected < 10ms`);
      assert.strictEqual(predictions.length, 100);
    });

    it('should predict 1000 samples in < 50ms', function() {
      const samples = generateLargeContinuousData(1000);
      
      const start = performance.now();
      const predictions = samples.map(sample => trainedModel.predict(sample));
      const duration = performance.now() - start;
      
      assert(duration < 50, `Batch prediction took ${duration}ms, expected < 50ms`);
      assert.strictEqual(predictions.length, 1000);
    });
  });
});

describe('Caching Performance Tests', function() {
  beforeEach(function() {
    globalCache.clear();
  });

  describe('Cache Hit Performance', function() {
    it('should improve performance with repeated predictions', function() {
      const data = generateLargeContinuousData(1000);
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      dt.train(data);
      
      const testSample = {
        feature1: 50,
        feature2: 25,
        feature3: 100
      };
      
      // First prediction (cold cache)
      const start1 = performance.now();
      const prediction1 = dt.predict(testSample);
      const coldDuration = performance.now() - start1;
      
      // Second prediction (warm cache)
      const start2 = performance.now();
      const prediction2 = dt.predict(testSample);
      const warmDuration = performance.now() - start2;
      
      assert.strictEqual(prediction1, prediction2);
      assert(warmDuration < coldDuration * 0.5, 
        `Warm prediction (${warmDuration}ms) should be < 50% of cold prediction (${coldDuration}ms)`);
    });

    it('should cache statistics effectively', function() {
      const data = generateLargeContinuousData(1000);
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      
      // First training (cold cache)
      const start1 = performance.now();
      dt.train(data);
      const coldDuration = performance.now() - start1;
      
      // Second training (warm cache)
      const start2 = performance.now();
      dt.train(data);
      const warmDuration = performance.now() - start2;
      
      assert(warmDuration < coldDuration * 0.7, 
        `Warm training should benefit from cached statistics`);
    });
  });

  describe('Cache Statistics', function() {
    it('should provide meaningful cache statistics', function() {
      const data = generateLargeContinuousData(1000);
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      dt.train(data);
      
      // Make some predictions
      const samples = generateLargeContinuousData(100);
      samples.forEach(sample => dt.predict(sample));
      
      const stats = dt.getCacheStats();
      assert(stats !== null);
      assert(typeof stats.predictionCache.size === 'number');
      assert(typeof stats.predictionCache.hitRate === 'number');
    });
  });
});

describe('Memory Usage Tests', function() {
  describe('Memory Footprint', function() {
    it('should use < 50MB for 100K samples', function() {
      const initialMemory = process.memoryUsage().heapUsed;
      const data = generateLargeContinuousData(100000);
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        memoryOptimization: true
      });
      dt.train(data);
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryUsed = (finalMemory - initialMemory) / 1024 / 1024; // MB
      
      assert(memoryUsed < 50, `Used ${memoryUsed}MB, expected < 50MB`);
    });

    it('should not leak memory during repeated training', function() {
      const data = generateLargeContinuousData(1000);
      
      for (let i = 0; i < 50; i++) {
        const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
          algorithm: 'cart',
          autoDetectTypes: true,
          memoryOptimization: true
        });
        dt.train(data);
        
        // Force garbage collection if available
        if (global.gc) global.gc();
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      // Memory should not grow significantly
      assert(finalMemory < 200 * 1024 * 1024, 'Memory leak detected');
    });
  });

  describe('Memory Optimization', function() {
    it('should optimize memory usage with large datasets', function() {
      const data = generateLargeContinuousData(10000);
      const features = ['feature1', 'feature2', 'feature3'];
      const target = 'target';
      const featureTypes = new Map([
        ['feature1', 'continuous'],
        ['feature2', 'continuous'],
        ['feature3', 'continuous']
      ]);
      
      const start = performance.now();
      const optimizedDataset = processDataOptimized(data, features, target, featureTypes);
      const duration = performance.now() - start;
      
      assert(duration < 100, `Memory optimization took ${duration}ms, expected < 100ms`);
      assert.strictEqual(optimizedDataset.sampleCount, 10000);
      assert(optimizedDataset.continuousFeatures.has('feature1'));
    });
  });
});

describe('Concurrent Performance Tests', function() {
  describe('Parallel Training', function() {
    it('should train multiple models in parallel efficiently', async function() {
      const datasets = Array(5).fill(null).map(() => 
        generateLargeContinuousData(1000));
      
      const start = performance.now();
      const promises = datasets.map(data => {
        const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
          algorithm: 'cart',
          autoDetectTypes: true
        });
        dt.train(data);
        return dt;
      });
      
      const models = await Promise.all(promises);
      const duration = performance.now() - start;
      
      assert(duration < 1000, `Parallel training took ${duration}ms, expected < 1000ms`);
      assert.strictEqual(models.length, 5);
    });

    it('should handle concurrent predictions efficiently', async function() {
      const data = generateLargeContinuousData(1000);
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        cachingEnabled: true
      });
      dt.train(data);
      
      const samples = generateLargeContinuousData(100);
      
      const start = performance.now();
      const promises = samples.map(sample => 
        new Promise(resolve => {
          setImmediate(() => resolve(dt.predict(sample)));
        })
      );
      
      await Promise.all(promises);
      const duration = performance.now() - start;
      
      assert(duration < 50, `Concurrent predictions took ${duration}ms, expected < 50ms`);
    });
  });
});

describe('Algorithm Performance Comparison', function() {
  describe('ID3 vs CART Performance', function() {
    it('should train faster on discrete data with ID3', function() {
      const discreteData = generateLargeMixedData(1000).map(row => ({
        discrete1: row.discrete1,
        discrete2: row.discrete2,
        target: row.target
      }));
      
      // ID3 on discrete data
      const start1 = performance.now();
      const dt1 = new DecisionTree('target', ['discrete1', 'discrete2'], {
        algorithm: 'id3',
        autoDetectTypes: true
      });
      dt1.train(discreteData);
      const id3Duration = performance.now() - start1;
      
      // CART on discrete data
      const start2 = performance.now();
      const dt2 = new DecisionTree('target', ['discrete1', 'discrete2'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt2.train(discreteData);
      const cartDuration = performance.now() - start2;
      
      assert(id3Duration < 50, `ID3 training too slow: ${id3Duration}ms`);
      assert(cartDuration < 100, `CART training too slow: ${cartDuration}ms`);
    });

    it('should train faster on continuous data with CART', function() {
      const continuousData = generateLargeContinuousData(1000);
      
      // CART on continuous data
      const start1 = performance.now();
      const dt1 = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'cart',
        autoDetectTypes: true
      });
      dt1.train(continuousData);
      const cartDuration = performance.now() - start1;
      
      // ID3 on continuous data (should fallback to discrete)
      const start2 = performance.now();
      const dt2 = new DecisionTree('target', ['feature1', 'feature2', 'feature3'], {
        algorithm: 'id3',
        autoDetectTypes: true
      });
      dt2.train(continuousData);
      const id3Duration = performance.now() - start2;
      
      assert(cartDuration < 50, `CART training too slow: ${cartDuration}ms`);
      assert(id3Duration < 100, `ID3 training too slow: ${id3Duration}ms`);
    });
  });

  describe('Hybrid Algorithm Performance', function() {
    it('should efficiently handle mixed data types', function() {
      const mixedData = generateLargeMixedData(1000);
      
      const start = performance.now();
      const dt = new DecisionTree('target', ['continuous1', 'continuous2', 'discrete1', 'discrete2'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      dt.train(mixedData);
      const duration = performance.now() - start;
      
      assert(duration < 75, `Hybrid training took ${duration}ms, expected < 75ms`);
    });
  });
});

describe('Edge Case Performance', function() {
  describe('High Cardinality Features', function() {
    it('should handle high-cardinality categorical features efficiently', function() {
      const highCardinalityData = generateLargeContinuousData(1000).map((row, i) => ({
        ...row,
        category: `category_${i % 50}` // 50 unique categories
      }));
      
      const start = performance.now();
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'category'], {
        algorithm: 'auto',
        autoDetectTypes: true
      });
      dt.train(highCardinalityData);
      const duration = performance.now() - start;
      
      assert(duration < 100, `High cardinality training took ${duration}ms, expected < 100ms`);
    });
  });

  describe('Sparse Data', function() {
    it('should handle sparse continuous features efficiently', function() {
      const sparseData = generateLargeContinuousData(1000).map(row => ({
        ...row,
        sparseFeature: Math.random() < 0.1 ? Math.random() * 100 : null // 90% missing values
      }));
      
      const start = performance.now();
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'sparseFeature'], {
        algorithm: 'cart',
        autoDetectTypes: true,
        handleMissingValues: true
      });
      dt.train(sparseData);
      const duration = performance.now() - start;
      
      assert(duration < 50, `Sparse data training took ${duration}ms, expected < 50ms`);
    });
  });
});
