import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('ID3 Algorithm Tests', () => {
  describe('Entropy and Information Gain', () => {
    it('should handle zero entropy datasets (all same target)', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const zeroEntropyData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class1' },
        { feature1: 'value3', target: 'class1' },
        { feature1: 'value4', target: 'class1' }
      ];
      
      dt.train(zeroEntropyData);
      // With zero entropy, any prediction should be 'class1'
      const prediction1 = dt.predict({ feature1: 'value1' });
      const prediction2 = dt.predict({ feature1: 'unknown' });
      assert.strictEqual(prediction1, 'class1');
      assert.strictEqual(prediction2, 'class1');
    });

    it('should handle maximum entropy datasets (perfect distribution)', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const maxEntropyData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' },
        { feature1: 'value3', target: 'class3' },
        { feature1: 'value4', target: 'class4' }
      ];
      
      dt.train(maxEntropyData);
      // Should still make predictions despite high entropy
      const prediction = dt.predict({ feature1: 'value1' });
      assert.ok(['class1', 'class2', 'class3', 'class4'].includes(prediction));
    });

    it('should handle balanced vs imbalanced datasets', () => {
      const dt = new DecisionTree('target', ['feature1']);
      
      // Balanced dataset
      const balancedData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      
      dt.train(balancedData);
      const balancedPrediction = dt.predict({ feature1: 'value1' });
      assert.strictEqual(balancedPrediction, 'class1');
      
      // Imbalanced dataset (90/10)
      const imbalancedData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      
      dt.train(imbalancedData);
      const imbalancedPrediction = dt.predict({ feature1: 'value1' });
      assert.strictEqual(imbalancedPrediction, 'class1');
    });
  });

  describe('Feature Selection and Splitting', () => {
    it('should handle single feature datasets', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const singleFeatureData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 'class2' }
      ];
      
      dt.train(singleFeatureData);
      const prediction1 = dt.predict({ feature1: 'value1' });
      const prediction2 = dt.predict({ feature1: 'value2' });
      assert.strictEqual(prediction1, 'class1');
      assert.strictEqual(prediction2, 'class2');
    });

    it('should handle categorical vs numerical features', () => {
      const dt = new DecisionTree('target', ['category', 'number']);
      const mixedData = [
        { category: 'A', number: 1, target: 'class1' },
        { category: 'A', number: 2, target: 'class1' },
        { category: 'B', number: 1, target: 'class2' },
        { category: 'B', number: 2, target: 'class2' }
      ];
      
      dt.train(mixedData);
      const prediction1 = dt.predict({ category: 'A', number: 1 });
      const prediction2 = dt.predict({ category: 'B', number: 2 });
      assert.strictEqual(prediction1, 'class1');
      assert.strictEqual(prediction2, 'class2');
    });

    it('should handle features with no predictive power', () => {
      const dt = new DecisionTree('target', ['useful', 'useless']);
      const data = [
        { useful: 'A', useless: 'X', target: 'class1' },
        { useful: 'A', useless: 'Y', target: 'class1' },
        { useful: 'B', useless: 'X', target: 'class2' },
        { useful: 'B', useless: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      // The tree should prioritize the 'useful' feature
      const prediction1 = dt.predict({ useful: 'A', useless: 'anything' });
      const prediction2 = dt.predict({ useful: 'B', useless: 'anything' });
      assert.strictEqual(prediction1, 'class1');
      assert.strictEqual(prediction2, 'class2');
    });
  });

  describe('Tree Structure and Depth', () => {
    it('should handle very deep tree structures', () => {
      const dt = new DecisionTree('target', ['level1', 'level2', 'level3', 'level4', 'level5']);
      const deepData = [
        { level1: 'A', level2: 'A', level3: 'A', level4: 'A', level5: 'A', target: 'class1' },
        { level1: 'A', level2: 'A', level3: 'A', level4: 'A', level5: 'B', target: 'class2' },
        { level1: 'A', level2: 'A', level3: 'A', level4: 'B', level5: 'A', target: 'class3' },
        { level1: 'A', level2: 'A', level3: 'A', level4: 'B', level5: 'B', target: 'class4' },
        { level1: 'A', level2: 'A', level3: 'B', level4: 'A', level5: 'A', target: 'class5' },
        { level1: 'A', level2: 'A', level3: 'B', level4: 'A', level5: 'B', target: 'class6' },
        { level1: 'A', level2: 'A', level3: 'B', level4: 'B', level5: 'A', target: 'class7' },
        { level1: 'A', level2: 'A', level3: 'B', level4: 'B', level5: 'B', target: 'class8' }
      ];
      
      dt.train(deepData);
      const prediction = dt.predict({ level1: 'A', level2: 'A', level3: 'A', level4: 'A', level5: 'A' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle wide tree structures (many feature values)', () => {
      const dt = new DecisionTree('target', ['feature']);
      const wideData = [];
      
      // Create data with 20 different feature values
      for (let i = 0; i < 20; i++) {
        wideData.push({ feature: `value${i}`, target: `class${i % 3 + 1}` });
      }
      
      dt.train(wideData);
      const prediction = dt.predict({ feature: 'value5' });
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });

    it('should handle balanced vs unbalanced tree splits', () => {
      const dt = new DecisionTree('target', ['feature']);
      
      // Balanced split
      const balancedData = [
        { feature: 'A', target: 'class1' },
        { feature: 'A', target: 'class1' },
        { feature: 'B', target: 'class2' },
        { feature: 'B', target: 'class2' }
      ];
      
      dt.train(balancedData);
      const balancedPrediction = dt.predict({ feature: 'A' });
      assert.strictEqual(balancedPrediction, 'class1');
      
      // Unbalanced split (3 vs 1)
      const unbalancedData = [
        { feature: 'A', target: 'class1' },
        { feature: 'A', target: 'class1' },
        { feature: 'A', target: 'class1' },
        { feature: 'B', target: 'class2' }
      ];
      
      dt.train(unbalancedData);
      const unbalancedPrediction = dt.predict({ feature: 'A' });
      assert.strictEqual(unbalancedPrediction, 'class1');
    });
  });

  describe('Algorithm Correctness', () => {
    it('should verify information gain calculations', () => {
      const dt = new DecisionTree('target', ['feature']);
      const data = [
        { feature: 'A', target: 'class1' },
        { feature: 'A', target: 'class1' },
        { feature: 'B', target: 'class2' },
        { feature: 'B', target: 'class2' }
      ];
      
      dt.train(data);
      const model = dt.toJSON();
      
      // Verify the tree structure makes sense
      assert.ok(model.model.type === 'feature' || model.model.type === 'result');
      if (model.model.type === 'feature') {
        assert.ok(model.model.vals);
        assert.ok(Array.isArray(model.model.vals));
        assert.ok(model.model.vals.length === 2); // Two feature values: A and B
      }
    });

    it('should handle tie-breaking in feature selection', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const tieData = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(tieData);
      // Both features should have equal information gain
      // The algorithm should handle this gracefully
      const prediction = dt.predict({ feature1: 'A', feature2: 'X' });
      assert.ok(['class1', 'class2'].includes(prediction));
    });

    it('should handle datasets with perfect correlation', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const correlatedData = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(correlatedData);
      // The tree should use one feature and ignore the other
      const prediction = dt.predict({ feature1: 'A', feature2: 'X' });
      assert.strictEqual(prediction, 'class1');
    });
  });
});
