import { strict as assert } from 'assert';
import RandomForest from '../lib/random-forest.js';

describe('Random Forest Edge Cases', function() {
  describe('Empty and Invalid Datasets', function() {
    it('should handle empty training dataset', () => {
      const rf = new RandomForest('target', ['feature1']);
      assert.throws(() => rf.train([]), /data.*Array.*Object/);
    });

    it('should handle null training dataset', () => {
      const rf = new RandomForest('target', ['feature1']);
      assert.throws(() => rf.train(null as any), /data.*Array.*Object/);
    });

    it('should handle undefined training dataset', () => {
      const rf = new RandomForest('target', ['feature1']);
      assert.throws(() => rf.train(undefined as any), /data.*Array.*Object/);
    });

    it('should handle non-array training dataset', () => {
      const rf = new RandomForest('target', ['feature1']);
      assert.throws(() => rf.train({} as any), /data.*Array.*Object/);
    });

    it('should handle single sample training dataset', () => {
      const singleSample = [{ feature1: 'value1', target: 'class1' }];
      const rf = new RandomForest('target', ['feature1'], { nEstimators: 5 });
      
      assert.doesNotThrow(() => rf.train(singleSample));
      assert.strictEqual(rf.getTreeCount(), 5);
    });

    it('should handle duplicate samples in training data', () => {
      const duplicateData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      const rf = new RandomForest('target', ['feature1'], { nEstimators: 5 });
      
      assert.doesNotThrow(() => rf.train(duplicateData));
      assert.strictEqual(rf.getTreeCount(), 5);
    });
  });

  describe('Missing Features and Data Inconsistencies', function() {
    it('should handle missing features in training data', () => {
      const inconsistentData = [
        { feature1: 'value1', target: 'class1' },
        { feature2: 'value2', target: 'class2' },
        { feature1: 'value3', feature2: 'value4', target: 'class1' }
      ];
      const rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 5 });
      
      assert.doesNotThrow(() => rf.train(inconsistentData));
      assert.strictEqual(rf.getTreeCount(), 5);
    });

    it('should handle samples with different feature sets', () => {
      const mixedData = [
        { feature1: 'value1', target: 'class1' },
        { feature2: 'value2', target: 'class2' },
        { feature1: 'value3', feature2: 'value4', extra: 'value', target: 'class1' }
      ];
      const rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 5 });
      
      assert.doesNotThrow(() => rf.train(mixedData));
      assert.strictEqual(rf.getTreeCount(), 5);
    });

    it('should handle null/undefined values in training data', () => {
      const nullData = [
        { feature1: 'value1', feature2: null, target: 'class1' },
        { feature1: null, feature2: 'value2', target: 'class2' },
        { feature1: 'value3', feature2: 'value4', target: 'class1' }
      ];
      const rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 5 });
      
      assert.doesNotThrow(() => rf.train(nullData));
      assert.strictEqual(rf.getTreeCount(), 5);
    });
  });

  describe('Constructor Edge Cases', function() {
    it('should handle empty features array', () => {
      const rf = new RandomForest('target', []);
      assert.doesNotThrow(() => rf.train([{ target: 'class1' }]));
    });

    it('should handle single feature', () => {
      const rf = new RandomForest('target', ['feature1']);
      const data = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle very long feature names', () => {
      const longFeatureName = 'a'.repeat(1000);
      const rf = new RandomForest('target', [longFeatureName]);
      const data = [
        { [longFeatureName]: 'value1', target: 'class1' },
        { [longFeatureName]: 'value2', target: 'class2' }
      ];
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle numeric feature names', () => {
      const rf = new RandomForest('target', ['1', '2', '3']);
      const data = [
        { '1': 'value1', '2': 'value2', '3': 'value3', target: 'class1' },
        { '1': 'value4', '2': 'value5', '3': 'value6', target: 'class2' }
      ];
      assert.doesNotThrow(() => rf.train(data));
    });
  });

  describe('Configuration Edge Cases', function() {
    it('should handle zero nEstimators', () => {
      const rf = new RandomForest('target', ['feature1'], { nEstimators: 0 });
      const data = [{ feature1: 'value1', target: 'class1' }];
      
      assert.doesNotThrow(() => rf.train(data));
      assert.strictEqual(rf.getTreeCount(), 0);
    });

  it('should handle negative nEstimators', () => {
    const rf = new RandomForest('target', ['feature1'], { nEstimators: -5 });
    const data = [{ feature1: 'value1', target: 'class1' }];
    
    assert.doesNotThrow(() => rf.train(data));
    // Negative nEstimators should result in 0 trees (loop doesn't run)
    assert.strictEqual(rf.getTreeCount(), 0);
  });

    it('should handle very large nEstimators', () => {
      const rf = new RandomForest('target', ['feature1'], { nEstimators: 10000 });
      const data = [{ feature1: 'value1', target: 'class1' }];
      
      assert.doesNotThrow(() => rf.train(data));
      assert.strictEqual(rf.getTreeCount(), 10000);
    });

    it('should handle maxFeatures larger than available features', () => {
      const rf = new RandomForest('target', ['feature1'], { maxFeatures: 10 });
      const data = [{ feature1: 'value1', target: 'class1' }];
      
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle negative maxFeatures', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { maxFeatures: -1 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle zero maxDepth', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { maxDepth: 0 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle negative maxDepth', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { maxDepth: -1 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle zero minSamplesSplit', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { minSamplesSplit: 0 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      assert.doesNotThrow(() => rf.train(data));
    });

    it('should handle negative minSamplesSplit', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { minSamplesSplit: -1 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      assert.doesNotThrow(() => rf.train(data));
    });
  });

  describe('Prediction Edge Cases', function() {
    let rf: RandomForest;
    const trainingData = [
      { feature1: 'value1', feature2: 'value2', target: 'class1' },
      { feature1: 'value3', feature2: 'value4', target: 'class2' },
      { feature1: 'value5', feature2: 'value6', target: 'class1' }
    ];

    beforeEach(() => {
      rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 5 });
      rf.train(trainingData);
    });

    it('should handle missing features in prediction sample', () => {
      const sample = { feature1: 'value1' }; // Missing feature2
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle completely empty prediction sample', () => {
      const sample = {};
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle sample with only some features', () => {
      const sample = { feature2: 'value2' }; // Only feature2
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle extra features in prediction sample', () => {
      const sample = { feature1: 'value1', feature2: 'value2', extra: 'value' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle null values in prediction sample', () => {
      const sample = { feature1: null, feature2: 'value2' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle undefined values in prediction sample', () => {
      const sample = { feature1: undefined, feature2: 'value2' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle unknown feature values', () => {
      const sample = { feature1: 'unknown', feature2: 'unknown' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle very long string values', () => {
      const longValue = 'a'.repeat(10000);
      const sample = { feature1: longValue, feature2: 'value2' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle special characters in feature values', () => {
      const sample = { feature1: '!@#$%^&*()', feature2: 'value2' };
      assert.doesNotThrow(() => rf.predict(sample));
    });

    it('should handle unicode characters in feature values', () => {
      const sample = { feature1: '🚀🎉💯', feature2: 'value2' };
      assert.doesNotThrow(() => rf.predict(sample));
    });
  });

  describe('Model Persistence Edge Cases', function() {
    it('should handle importing empty model', () => {
      const emptyModel = {
        trees: [],
        target: 'target',
        features: ['feature1'],
        config: { nEstimators: 0 },
        data: []
      };
      
      const rf = new RandomForest(emptyModel);
      assert.strictEqual(rf.getTreeCount(), 0);
    });

    it('should handle importing model with missing properties', () => {
      const incompleteModel = {
        trees: [],
        target: 'target'
        // Missing features, config, data
      } as any;
      
      assert.throws(() => new RandomForest(incompleteModel));
    });

    it('should handle importing model with corrupted tree data', () => {
      const corruptedModel = {
        trees: [{ model: null, data: [], target: 'target', features: ['feature1'] }],
        target: 'target',
        features: ['feature1'],
        config: { nEstimators: 1 },
        data: []
      };
      
      // This should not throw an error during import, but should fail during prediction
      const rf = new RandomForest(corruptedModel);
      assert.throws(() => rf.predict({ feature1: 'value1' }));
    });

    it('should handle round-trip export/import with edge case data', () => {
      const edgeCaseData = [
        { feature1: null, feature2: undefined, target: 'class1' },
        { feature1: '', feature2: 'value2', target: 'class2' },
        { feature1: 'value3', feature2: '', target: 'class1' }
      ];
      
      const rf1 = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 3 });
      rf1.train(edgeCaseData);
      
      const modelJson = rf1.toJSON();
      const rf2 = new RandomForest(modelJson);
      
      assert.strictEqual(rf2.getTreeCount(), 3);
      
      // Test prediction consistency
      const sample = { feature1: 'test', feature2: 'test' };
      const pred1 = rf1.predict(sample);
      const pred2 = rf2.predict(sample);
      assert.strictEqual(pred1, pred2);
    });
  });

  describe('Feature Importance Edge Cases', function() {
    it('should handle feature importance with single tree', () => {
      const rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 1 });
      const data = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      rf.train(data);
      const importance = rf.getFeatureImportance();
      
      assert.ok(typeof importance === 'object');
      assert.ok('feature1' in importance);
      assert.ok('feature2' in importance);
    });

  it('should handle feature importance with zero trees', () => {
    const rf = new RandomForest('target', ['feature1', 'feature2'], { nEstimators: 0 });
    const data = [{ feature1: 'value1', feature2: 'value2', target: 'class1' }];
    
    rf.train(data);
    // With zero trees, getFeatureImportance should throw an error
    assert.throws(() => rf.getFeatureImportance(), /Random Forest has not been trained yet/);
  });

    it('should handle feature importance with single feature', () => {
      const rf = new RandomForest('target', ['feature1'], { nEstimators: 5 });
      const data = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      
      rf.train(data);
      const importance = rf.getFeatureImportance();
      
      assert.ok(typeof importance === 'object');
      assert.ok('feature1' in importance);
      assert.ok(importance.feature1 >= 0);
    });
  });
});
