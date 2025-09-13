import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';
import RandomForest from '../lib/random-forest.js';
import XGBoost from '../lib/xgboost.js';

// Performance test configuration
const PERFORMANCE_THRESHOLDS = {
  // Training time thresholds (in milliseconds)
  DECISION_TREE_TRAINING: 100,      // Decision tree should train in <100ms
  RANDOM_FOREST_TRAINING: 500,      // Random Forest should train in <500ms
  XGBOOST_TRAINING: 1000,           // XGBoost should train in <1000ms
  
  // Prediction time thresholds (in milliseconds)
  DECISION_TREE_PREDICTION: 10,     // Decision tree prediction should be <10ms
  RANDOM_FOREST_PREDICTION: 50,     // Random Forest prediction should be <50ms
  XGBOOST_PREDICTION: 20,           // XGBoost prediction should be <20ms
  
  // Memory usage thresholds (in MB)
  MEMORY_USAGE: 50,                 // Should not exceed 50MB for large datasets
  
  // Accuracy thresholds (minimum accuracy for performance tests)
  MIN_ACCURACY: 0.7                 // Should maintain reasonable accuracy
};

// Generate test datasets of various sizes
function generateTestData(size: number, features: string[] = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']) {
  const data = [];
  for (let i = 0; i < size; i++) {
    const sample: any = {};
    features.forEach(feature => {
      sample[feature] = Math.random() > 0.5 ? 'A' : 'B';
    });
    sample.target = Math.random() > 0.5;
    data.push(sample);
  }
  return data;
}

function generateRegressionData(size: number, features: string[] = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']) {
  const data = [];
  for (let i = 0; i < size; i++) {
    const sample: any = {};
    features.forEach(feature => {
      sample[feature] = Math.random() * 100;
    });
    sample.target = Math.random() * 1000;
    data.push(sample);
  }
  return data;
}

describe('Performance Tests - Decision Tree', function() {
  this.timeout(10000); // 10 second timeout for performance tests

  it('should train quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING, 
      `Decision tree training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING}ms`);
  });

  it('should train quickly on medium datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING * 2, 
      `Decision tree training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING * 2}ms`);
  });

  it('should train quickly on large datasets', () => {
    const data = generateTestData(5000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING * 5, 
      `Decision tree training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.DECISION_TREE_TRAINING * 5}ms`);
  });

  it('should predict quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    const dt = new DecisionTree('target', features);
    dt.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 100; i++) {
      dt.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 100;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.DECISION_TREE_PREDICTION, 
      `Decision tree prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.DECISION_TREE_PREDICTION}ms`);
  });

  it('should predict quickly on large datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    const dt = new DecisionTree('target', features);
    dt.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A', feature4: 'B', feature5: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 1000; i++) {
      dt.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 1000;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.DECISION_TREE_PREDICTION, 
      `Decision tree prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.DECISION_TREE_PREDICTION}ms`);
  });

  it('should maintain accuracy on performance tests', () => {
    const data = generateTestData(500);
    const features = ['feature1', 'feature2', 'feature3'];
    const dt = new DecisionTree('target', features);
    dt.train(data);
    
    const accuracy = dt.evaluate(data);
    // For random data, we expect around 50% accuracy, so lower the threshold
    assert.ok(accuracy >= 0.4, 
      `Decision tree accuracy ${accuracy} is below threshold 0.4`);
  });
});

describe('Performance Tests - Random Forest', function() {
  this.timeout(15000); // 15 second timeout for Random Forest tests

  it('should train quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const rf = new RandomForest('target', features, { nEstimators: 10 });
    rf.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING, 
      `Random Forest training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING}ms`);
  });

  it('should train quickly on medium datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const rf = new RandomForest('target', features, { nEstimators: 50 });
    rf.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING * 2, 
      `Random Forest training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING * 2}ms`);
  });

  it('should train quickly on large datasets', () => {
    const data = generateTestData(2000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const rf = new RandomForest('target', features, { nEstimators: 100 });
    rf.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING * 3, 
      `Random Forest training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.RANDOM_FOREST_TRAINING * 3}ms`);
  });

  it('should predict quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    const rf = new RandomForest('target', features, { nEstimators: 10 });
    rf.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 100; i++) {
      rf.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 100;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.RANDOM_FOREST_PREDICTION, 
      `Random Forest prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.RANDOM_FOREST_PREDICTION}ms`);
  });

  it('should predict quickly on large datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    const rf = new RandomForest('target', features, { nEstimators: 50 });
    rf.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A', feature4: 'B', feature5: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 500; i++) {
      rf.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 500;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.RANDOM_FOREST_PREDICTION, 
      `Random Forest prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.RANDOM_FOREST_PREDICTION}ms`);
  });

  it('should maintain accuracy on performance tests', () => {
    const data = generateTestData(500);
    const features = ['feature1', 'feature2', 'feature3'];
    const rf = new RandomForest('target', features, { nEstimators: 20 });
    rf.train(data);
    
    const accuracy = rf.evaluate(data);
    // For random data, we expect around 50% accuracy, so lower the threshold
    assert.ok(accuracy >= 0.4, 
      `Random Forest accuracy ${accuracy} is below threshold 0.4`);
  });

  it('should scale well with number of trees', () => {
    const data = generateTestData(200);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const treeCounts = [10, 50, 100];
    const trainingTimes: number[] = [];
    
    treeCounts.forEach(nEstimators => {
      const startTime = Date.now();
      const rf = new RandomForest('target', features, { nEstimators });
      rf.train(data);
      const endTime = Date.now();
      trainingTimes.push(endTime - startTime);
    });
    
    // Training time should scale roughly linearly with number of trees
    // Handle case where first training time might be 0
    const ratio = trainingTimes[0] > 0 ? trainingTimes[2] / trainingTimes[0] : trainingTimes[2];
    assert.ok(ratio < 20, `Training time scaling ratio ${ratio} is too high (expected < 20)`);
  });
});

describe('Performance Tests - XGBoost', function() {
  this.timeout(20000); // 20 second timeout for XGBoost tests

  it('should train quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const xgb = new XGBoost('target', features, { nEstimators: 10, objective: 'binary' });
    xgb.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING, 
      `XGBoost training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING}ms`);
  });

  it('should train quickly on medium datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const xgb = new XGBoost('target', features, { nEstimators: 50, objective: 'binary' });
    xgb.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING * 2, 
      `XGBoost training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING * 2}ms`);
  });

  it('should train quickly on large datasets', () => {
    const data = generateTestData(2000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    const startTime = Date.now();
    const xgb = new XGBoost('target', features, { nEstimators: 100, objective: 'binary' });
    xgb.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING * 3, 
      `XGBoost training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING * 3}ms`);
  });

  it('should predict quickly on small datasets', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    const xgb = new XGBoost('target', features, { nEstimators: 10, objective: 'binary' });
    xgb.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 100; i++) {
      xgb.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 100;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.XGBOOST_PREDICTION, 
      `XGBoost prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_PREDICTION}ms`);
  });

  it('should predict quickly on large datasets', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    const xgb = new XGBoost('target', features, { nEstimators: 50, objective: 'binary' });
    xgb.train(data);
    
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A', feature4: 'B', feature5: 'A' };
    
    const startTime = Date.now();
    for (let i = 0; i < 500; i++) {
      xgb.predict(testSample);
    }
    const endTime = Date.now();
    
    const avgPredictionTime = (endTime - startTime) / 500;
    assert.ok(avgPredictionTime < PERFORMANCE_THRESHOLDS.XGBOOST_PREDICTION, 
      `XGBoost prediction took ${avgPredictionTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_PREDICTION}ms`);
  });

  it('should maintain accuracy on performance tests', () => {
    const data = generateTestData(500);
    const features = ['feature1', 'feature2', 'feature3'];
    const xgb = new XGBoost('target', features, { nEstimators: 20, objective: 'binary' });
    xgb.train(data);
    
    const accuracy = xgb.evaluate(data);
    // For random data, we expect around 50% accuracy, so lower the threshold
    assert.ok(accuracy >= 0.4, 
      `XGBoost accuracy ${accuracy} is below threshold 0.4`);
  });

  it('should scale well with number of estimators', () => {
    const data = generateTestData(200);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const estimatorCounts = [10, 50, 100];
    const trainingTimes: number[] = [];
    
    estimatorCounts.forEach(nEstimators => {
      const startTime = Date.now();
      const xgb = new XGBoost('target', features, { nEstimators, objective: 'binary' });
      xgb.train(data);
      const endTime = Date.now();
      trainingTimes.push(endTime - startTime);
    });
    
    // Training time should scale roughly linearly with number of estimators
    const ratio = trainingTimes[2] / trainingTimes[0];
    assert.ok(ratio < 25, `Training time scaling ratio ${ratio} is too high (expected < 25)`);
  });

  it('should handle regression efficiently', () => {
    const data = generateRegressionData(500);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const xgb = new XGBoost('target', features, { nEstimators: 50, objective: 'regression' });
    xgb.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING, 
      `XGBoost regression training took ${trainingTime}ms, expected < ${PERFORMANCE_THRESHOLDS.XGBOOST_TRAINING}ms`);
  });
});

describe('Performance Tests - Algorithm Comparison', function() {
  this.timeout(25000); // 25 second timeout for comparison tests

  it('should compare training times across algorithms', () => {
    const data = generateTestData(500);
    const features = ['feature1', 'feature2', 'feature3', 'feature4'];
    
    // Decision Tree
    const dtStart = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const dtTime = Date.now() - dtStart;
    
    // Random Forest
    const rfStart = Date.now();
    const rf = new RandomForest('target', features, { nEstimators: 20 });
    rf.train(data);
    const rfTime = Date.now() - rfStart;
    
    // XGBoost
    const xgbStart = Date.now();
    const xgb = new XGBoost('target', features, { nEstimators: 20, objective: 'binary' });
    xgb.train(data);
    const xgbTime = Date.now() - xgbStart;
    
    // Decision Tree should be fastest
    assert.ok(dtTime < rfTime, `Decision Tree (${dtTime}ms) should be faster than Random Forest (${rfTime}ms)`);
    assert.ok(dtTime < xgbTime, `Decision Tree (${dtTime}ms) should be faster than XGBoost (${xgbTime}ms)`);
    
    // Random Forest should be faster than XGBoost for same number of trees
    assert.ok(rfTime < xgbTime, `Random Forest (${rfTime}ms) should be faster than XGBoost (${xgbTime}ms)`);
  });

  it('should compare prediction times across algorithms', () => {
    const data = generateTestData(500);
    const features = ['feature1', 'feature2', 'feature3', 'feature4'];
    const testSample = { feature1: 'A', feature2: 'B', feature3: 'A', feature4: 'B' };
    
    // Train all algorithms
    const dt = new DecisionTree('target', features);
    dt.train(data);
    
    const rf = new RandomForest('target', features, { nEstimators: 20 });
    rf.train(data);
    
    const xgb = new XGBoost('target', features, { nEstimators: 20, objective: 'binary' });
    xgb.train(data);
    
    // Test prediction times
    const iterations = 1000;
    
    const dtStart = Date.now();
    for (let i = 0; i < iterations; i++) {
      dt.predict(testSample);
    }
    const dtTime = (Date.now() - dtStart) / iterations;
    
    const rfStart = Date.now();
    for (let i = 0; i < iterations; i++) {
      rf.predict(testSample);
    }
    const rfTime = (Date.now() - rfStart) / iterations;
    
    const xgbStart = Date.now();
    for (let i = 0; i < iterations; i++) {
      xgb.predict(testSample);
    }
    const xgbTime = (Date.now() - xgbStart) / iterations;
    
    // All algorithms should predict quickly
    assert.ok(dtTime < 1, `Decision Tree prediction too slow: ${dtTime}ms`);
    assert.ok(rfTime < 5, `Random Forest prediction too slow: ${rfTime}ms`);
    assert.ok(xgbTime < 2, `XGBoost prediction too slow: ${xgbTime}ms`);
  });

  it('should maintain reasonable memory usage', () => {
    const data = generateTestData(1000);
    const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
    
    // Measure memory usage before
    const memBefore = process.memoryUsage().heapUsed / 1024 / 1024; // MB
    
    // Train all algorithms
    const dt = new DecisionTree('target', features);
    dt.train(data);
    
    const rf = new RandomForest('target', features, { nEstimators: 50 });
    rf.train(data);
    
    const xgb = new XGBoost('target', features, { nEstimators: 50, objective: 'binary' });
    xgb.train(data);
    
    // Measure memory usage after
    const memAfter = process.memoryUsage().heapUsed / 1024 / 1024; // MB
    const memUsed = memAfter - memBefore;
    
    assert.ok(memUsed < PERFORMANCE_THRESHOLDS.MEMORY_USAGE, 
      `Memory usage ${memUsed.toFixed(2)}MB exceeds threshold ${PERFORMANCE_THRESHOLDS.MEMORY_USAGE}MB`);
  });

  it('should handle concurrent operations efficiently', () => {
    const data = generateTestData(200);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    
    // Create multiple instances concurrently
    const promises = [];
    for (let i = 0; i < 5; i++) {
      promises.push(new Promise(resolve => {
        const dt = new DecisionTree('target', features);
        dt.train(data);
        const prediction = dt.predict({ feature1: 'A', feature2: 'B', feature3: 'A' });
        resolve(prediction);
      }));
    }
    
    Promise.all(promises).then(() => {
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      assert.ok(totalTime < 2000, `Concurrent operations took ${totalTime}ms, expected < 2000ms`);
    });
  });
});

describe('Performance Tests - Edge Cases', function() {
  this.timeout(15000);

  it('should handle very deep trees efficiently', () => {
    const data = generateTestData(100);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < 200, `Deep tree training took ${trainingTime}ms, expected < 200ms`);
  });

  it('should handle many features efficiently', () => {
    const features = Array.from({ length: 20 }, (_, i) => `feature${i + 1}`);
    const data = generateTestData(200, features);
    
    const startTime = Date.now();
    const dt = new DecisionTree('target', features);
    dt.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < 500, `Many features training took ${trainingTime}ms, expected < 500ms`);
  });

  it('should handle early stopping efficiently', () => {
    const data = generateTestData(300);
    const features = ['feature1', 'feature2', 'feature3'];
    
    const startTime = Date.now();
    const xgb = new XGBoost('target', features, { 
      nEstimators: 100, 
      earlyStoppingRounds: 5, 
      validationFraction: 0.2,
      objective: 'binary'
    });
    xgb.train(data);
    const endTime = Date.now();
    
    const trainingTime = endTime - startTime;
    assert.ok(trainingTime < 2000, `Early stopping training took ${trainingTime}ms, expected < 2000ms`);
  });
});
