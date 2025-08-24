import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Edge Cases & Error Handling', () => {
  describe('Empty and Invalid Datasets', () => {
    it('should handle empty training dataset', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      // Note: Current implementation only validates that data is an array, not that it has elements
      // Empty arrays are allowed and will create a tree with no features
      dt.train([]);
      assert.ok(dt.toJSON());
    });

    it('should handle single sample training dataset', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const singleSample = [{ feature1: 'value1', feature2: 'value2', target: 'class1' }];
      
      dt.train(singleSample);
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle null/undefined values in training data', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const dataWithNulls = [
        { feature1: 'value1', feature2: null, target: 'class1' },
        { feature1: 'value1', feature2: 'value2', target: 'class2' },
        { feature1: undefined, feature2: 'value2', target: 'class1' }
      ];
      
      dt.train(dataWithNulls);
      // Should not crash and should make predictions
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.ok(typeof prediction === 'string');
    });

    it('should handle duplicate feature values', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const duplicateData = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value1', feature2: 'value2', target: 'class1' }
      ];
      
      dt.train(duplicateData);
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.strictEqual(prediction, 'class1');
    });
  });

  describe('Missing Features and Data Inconsistencies', () => {
    it('should handle missing features in training data', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3']);
      const incompleteData = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value1', feature2: 'value2', feature3: 'value3', target: 'class2' },
        { feature1: 'value1', target: 'class1' }
      ];
      
      dt.train(incompleteData);
      // Should handle missing features gracefully
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.ok(typeof prediction === 'string');
    });

    it('should handle data consistency across samples', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const inconsistentData = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value1', feature2: 'value2', extraFeature: 'extra', target: 'class2' },
        { feature1: 'value1', feature2: 'value2', target: 'class1' }
      ];
      
      dt.train(inconsistentData);
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Constructor Edge Cases', () => {
    it('should handle empty features array', () => {
      const dt = new DecisionTree('target', []);
      const data = [{ target: 'class1' }];
      
      dt.train(data);
      const prediction = dt.predict({});
      assert.ok(typeof prediction === 'string');
    });

    it('should handle single feature', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const data = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      
      dt.train(data);
      const prediction = dt.predict({ feature1: 'value1' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle very long feature names', () => {
      const longFeatureName = 'a'.repeat(1000);
      const dt = new DecisionTree('target', [longFeatureName]);
      const data = [{ [longFeatureName]: 'value1', target: 'class1' }];
      
      dt.train(data);
      const prediction = dt.predict({ [longFeatureName]: 'value1' });
      assert.strictEqual(prediction, 'class1');
    });
  });
});
