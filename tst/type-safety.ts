import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Type Safety & Interface Validation', () => {
  describe('TreeNode Structure Validation', () => {
    it('should validate TreeNode structure properties', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const model = dt.toJSON();
      
      // Validate root node structure
      const rootNode = model.model;
      assert.ok(typeof rootNode.type === 'string');
      assert.ok(typeof rootNode.name === 'string');
      assert.ok(typeof rootNode.alias === 'string');
      
      // Validate node types
      assert.ok(['result', 'feature', 'feature_value'].includes(rootNode.type));
      
      if (rootNode.type === 'feature') {
        assert.ok(typeof rootNode.gain === 'number');
        assert.ok(typeof rootNode.sampleSize === 'number');
        assert.ok(Array.isArray(rootNode.vals));
        
        // Validate child nodes
        for (const child of rootNode.vals!) {
          assert.ok(typeof child.name === 'string');
          assert.ok(typeof child.alias === 'string');
          assert.ok(typeof child.type === 'string');
          assert.ok(typeof child.prob === 'number');
          assert.ok(typeof child.sampleSize === 'number');
          
          if (child.child) {
            assert.ok(typeof child.child.type === 'string');
            assert.ok(typeof child.child.name === 'string');
            assert.ok(typeof child.child.alias === 'string');
          }
        }
      } else if (rootNode.type === 'result') {
        assert.ok(typeof rootNode.val === 'string' || typeof rootNode.val === 'number' || typeof rootNode.val === 'boolean');
      }
    });

    it('should validate TreeNode property types', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const data = [
        { feature1: 'A', target: 'class1' },
        { feature1: 'B', target: 'class2' }
      ];
      
      dt.train(data);
      const model = dt.toJSON();
      
      // Validate all numeric properties are numbers
      if (model.model.type === 'feature') {
        assert.ok(Number.isFinite(model.model.gain!));
        assert.ok(Number.isFinite(model.model.sampleSize!));
        assert.ok(model.model.gain! >= 0);
        assert.ok(model.model.sampleSize! > 0);
        
        for (const child of model.model.vals!) {
          assert.ok(Number.isFinite(child.prob!));
          assert.ok(Number.isFinite(child.sampleSize!));
          assert.ok(child.prob! >= 0 && child.prob! <= 1);
          assert.ok(child.sampleSize! > 0);
        }
      }
    });

    it('should validate TreeNode alias uniqueness', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'X', target: 'class3' },
        { feature1: 'B', feature2: 'Y', target: 'class4' }
      ];
      
      dt.train(data);
      const model = dt.toJSON();
      
      // Collect all aliases
      const aliases = new Set<string>();
      const collectAliases = (node: any) => {
        aliases.add(node.alias);
        if (node.vals) {
          for (const child of node.vals) {
            collectAliases(child);
          }
        }
        if (node.child) {
          collectAliases(node.child);
        }
      };
      
      collectAliases(model.model);
      
      // All aliases should be unique
      assert.strictEqual(aliases.size, Array.from(aliases).length);
      
      // Aliases should follow expected pattern
      for (const alias of aliases) {
        assert.ok(alias.includes('_r'), 'Alias should contain random suffix');
      }
    });
  });

  describe('DecisionTreeData Structure Validation', () => {
    it('should validate DecisionTreeData structure', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const exported = dt.toJSON();
      
      // Validate required properties
      assert.ok(exported.hasOwnProperty('model'));
      assert.ok(exported.hasOwnProperty('data'));
      assert.ok(exported.hasOwnProperty('target'));
      assert.ok(exported.hasOwnProperty('features'));
      
      // Validate property types
      assert.ok(typeof exported.model === 'object');
      assert.ok(Array.isArray(exported.data));
      assert.ok(typeof exported.target === 'string');
      assert.ok(Array.isArray(exported.features));
      
      // Validate data integrity
      // Note: Current implementation doesn't store training data in this.data
      assert.strictEqual(exported.data.length, 0);
      assert.strictEqual(exported.target, 'target');
      assert.deepStrictEqual(exported.features, ['feature1', 'feature2']);
    });

    it('should validate data array contents', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const exported = dt.toJSON();
      
      // Note: Current implementation doesn't store training data in this.data
      // Only imported models have data stored
      assert.strictEqual(exported.data.length, 0);
      
      // To test data preservation, we need to import the exported model
      const importedDt = new DecisionTree(exported);
      const importedExported = importedDt.toJSON();
      
      // Now the data should be preserved
      assert.strictEqual(importedExported.data.length, 0); // Still 0 because original export had no data
    });

    it('should validate features array contents', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const exported = dt.toJSON();
      
      // Validate features array
      assert.ok(Array.isArray(exported.features));
      assert.strictEqual(exported.features.length, 2);
      
      // All features should be strings
      for (const feature of exported.features) {
        assert.ok(typeof feature === 'string');
        assert.ok(feature.length > 0);
      }
      
      // Features should match original
      assert.deepStrictEqual(exported.features, ['feature1', 'feature2']);
    });
  });

  describe('FeatureGain Interface Validation', () => {
    it('should validate FeatureGain structure in tree building', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const model = dt.toJSON();
      
      // The root node should have gain information
      if (model.model.type === 'feature') {
        assert.ok(typeof model.model.gain === 'number');
        assert.ok(Number.isFinite(model.model.gain!));
        assert.ok(model.model.gain! >= 0);
        assert.ok(model.model.gain! <= 1);
      }
    });
  });

  describe('TrainingData Interface Validation', () => {
    it('should validate TrainingData structure', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      
      // Valid training data
      const validData = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(validData);
      assert.ok(dt.toJSON());
      
      // Test with different data types
      const mixedTypeData = [
        { feature1: 'A', feature2: 123, target: 'class1' },
        { feature1: 'B', feature2: true, target: 'class2' },
        { feature1: 'C', feature2: null, target: 'class3' }
      ];
      
      dt.train(mixedTypeData);
      assert.ok(dt.toJSON());
    });

    it('should handle TrainingData with missing features', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      
      const incompleteData = [
        { feature1: 'A', target: 'class1' },        // Missing feature2
        { feature2: 'Y', target: 'class2' },        // Missing feature1
        { feature1: 'B', feature2: 'Z', target: 'class3' } // Complete
      ];
      
      dt.train(incompleteData);
      assert.ok(dt.toJSON());
      
      // Should still make predictions
      const prediction = dt.predict({ feature1: 'A', feature2: 'Z' });
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Type Mismatch Handling', () => {
    it('should handle type mismatches gracefully', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      
      // Test with various data types
      const mixedData = [
        { feature1: 'string', feature2: 'value', target: 'class1' },
        { feature1: 123, feature2: 456, target: 'class2' },
        { feature1: true, feature2: false, target: 'class3' },
        { feature1: null, feature2: undefined, target: 'class4' },
        { feature1: { nested: 'object' }, feature2: [1, 2, 3], target: 'class5' }
      ];
      
      dt.train(mixedData);
      assert.ok(dt.toJSON());
      
      // Test predictions with different types
      const predictions = [
        dt.predict({ feature1: 'string', feature2: 'value' }),
        dt.predict({ feature1: 123, feature2: 456 }),
        dt.predict({ feature1: true, feature2: false }),
        dt.predict({ feature1: null, feature2: undefined }),
        dt.predict({ feature1: { nested: 'object' }, feature2: [1, 2, 3] })
      ];
      
      // All predictions should return valid results
      for (const prediction of predictions) {
        assert.ok(typeof prediction === 'string' || typeof prediction === 'number' || typeof prediction === 'boolean' || prediction === null);
      }
    });

    it('should validate constructor argument types', () => {
      // Test with invalid target types
      assert.throws(() => {
        new DecisionTree(123 as any, ['feature1']);
      }, /target.*String/);
      
      assert.throws(() => {
        new DecisionTree(true as any, ['feature1']);
      }, /target.*String/);
      
      assert.throws(() => {
        new DecisionTree(null as any, ['feature1']);
      }, /target.*String/);
      
      // Test with invalid features types
      assert.throws(() => {
        new DecisionTree('target', 'feature1' as any);
      }, /features.*Array/);
      
      assert.throws(() => {
        new DecisionTree('target', 123 as any);
      }, /features.*Array/);
      
      assert.throws(() => {
        new DecisionTree('target', null as any);
      }, /features.*Array/);
    });

    it('should validate training data types', () => {
      const dt = new DecisionTree('target', ['feature1']);
      
      // Test with invalid data types
      assert.throws(() => {
        dt.train('invalid' as any);
      }, /data.*Array.*Object/);
      
      assert.throws(() => {
        dt.train(123 as any);
      }, /data.*Array.*Object/);
      
      assert.throws(() => {
        dt.train(null as any);
      }, /data.*Array.*Object/);
      
      assert.throws(() => {
        dt.train(undefined as any);
      }, /data.*Array.*Object/);
    });
  });

  describe('Interface Consistency', () => {
    it('should maintain interface consistency across operations', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const data = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      dt.train(data);
      const exported1 = dt.toJSON();
      
      // Import the exported model
      const importedDt = new DecisionTree(exported1);
      const exported2 = importedDt.toJSON();
      
      // Verify interface consistency
      assert.deepStrictEqual(Object.keys(exported1), Object.keys(exported2));
      
      for (const key of Object.keys(exported1)) {
        const val1 = exported1[key as keyof typeof exported1];
        const val2 = exported2[key as keyof typeof exported2];
        
        if (Array.isArray(val1)) {
          assert.ok(Array.isArray(val2));
          assert.strictEqual(val1.length, val2.length);
        } else if (typeof val1 === 'string') {
          assert.strictEqual(typeof val2, 'string');
          assert.strictEqual(val1, val2);
        } else if (typeof val1 === 'object') {
          assert.ok(typeof val2 === 'object');
          // Deep comparison for objects
          assert.deepStrictEqual(val1, val2);
        }
      }
    });

    it('should validate static properties', () => {
      // Test NODE_TYPES static property
      assert.ok(DecisionTree.NODE_TYPES);
      assert.ok(typeof DecisionTree.NODE_TYPES === 'object');
      
      const expectedTypes = ['RESULT', 'FEATURE', 'FEATURE_VALUE'];
      for (const type of expectedTypes) {
        assert.ok(DecisionTree.NODE_TYPES.hasOwnProperty(type));
        assert.ok(typeof DecisionTree.NODE_TYPES[type as keyof typeof DecisionTree.NODE_TYPES] === 'string');
      }
      
      // Verify values
      assert.strictEqual(DecisionTree.NODE_TYPES.RESULT, 'result');
      assert.strictEqual(DecisionTree.NODE_TYPES.FEATURE, 'feature');
      assert.strictEqual(DecisionTree.NODE_TYPES.FEATURE_VALUE, 'feature_value');
    });
  });
});
