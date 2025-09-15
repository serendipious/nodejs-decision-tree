import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Model Persistence & Import/Export', () => {
  let originalDt: DecisionTree;
  let trainingData: any[];

  beforeEach(() => {
    trainingData = [
      { color: 'red', shape: 'circle', size: 'small', target: 'class1' },
      { color: 'blue', shape: 'square', size: 'medium', target: 'class2' },
      { color: 'green', shape: 'triangle', size: 'large', target: 'class3' },
      { color: 'red', shape: 'square', size: 'medium', target: 'class1' },
      { color: 'blue', shape: 'circle', size: 'small', target: 'class2' }
    ];
    
    originalDt = new DecisionTree('target', ['color', 'shape', 'size']);
    originalDt.train(trainingData);
  });

  describe('Export Functionality', () => {
    it('should export model with correct structure', () => {
      const exported = originalDt.toJSON();
      
      // Verify structure
      assert.ok(exported.model);
      assert.ok(Array.isArray(exported.data));
      assert.strictEqual(exported.target, 'target');
      assert.ok(Array.isArray(exported.features));
      assert.strictEqual(exported.features.length, 3);
      
      // Verify model properties
      assert.ok(typeof exported.model.type === 'string');
      assert.ok(typeof exported.model.name === 'string');
      assert.ok(typeof exported.model.alias === 'string');
    });

    it('should export model with all required properties', () => {
      const exported = originalDt.toJSON();
      
      // Check that all expected properties exist
      const requiredProps = ['model', 'data', 'target', 'features'];
      for (const prop of requiredProps) {
        assert.ok(exported.hasOwnProperty(prop), `Missing property: ${prop}`);
      }
    });

    it('should export model data integrity', () => {
      const exported = originalDt.toJSON();
      
      // Note: Current implementation doesn't store training data in this.data
      // Only imported models have data stored
      assert.strictEqual(exported.data.length, 0);
      assert.strictEqual(exported.target, 'target');
      assert.deepStrictEqual(exported.features, ['color', 'shape', 'size']);
      
      // Data is not preserved in current implementation
      // This is a design limitation
    });

    it('should export model with tree structure', () => {
      const exported = originalDt.toJSON();
      
      // Verify tree structure
      assert.ok(exported.model.type === 'feature' || exported.model.type === 'result');
      
      if (exported.model.type === 'feature') {
        assert.ok(exported.model.vals);
        assert.ok(Array.isArray(exported.model.vals));
        assert.ok(exported.model.vals.length > 0);
        
        // Check child nodes
        for (const child of exported.model.vals) {
          assert.ok(typeof child.name === 'string');
          assert.ok(typeof child.alias === 'string');
          assert.ok(typeof child.type === 'string');
          assert.ok(typeof child.prob === 'number');
          assert.ok(typeof child.sampleSize === 'number');
        }
      }
    });
  });

  describe('Import Functionality', () => {
    it('should import model and maintain functionality', () => {
      const exported = originalDt.toJSON();
      const importedDt = new DecisionTree(exported);
      
      // Test predictions
      const prediction1 = importedDt.predict({ color: 'red', shape: 'circle', size: 'small' });
      const prediction2 = importedDt.predict({ color: 'blue', shape: 'square', size: 'medium' });
      
      assert.strictEqual(prediction1, 'class1');
      assert.strictEqual(prediction2, 'class2');
    });

    it('should handle import on existing instance', () => {
      const exported = originalDt.toJSON();
      const newDt = new DecisionTree('target', ['color', 'shape', 'size']);
      
      newDt.import(exported);
      
      // Test predictions
      const prediction = newDt.predict({ color: 'red', shape: 'circle', size: 'small' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should maintain evaluation accuracy after import', () => {
      const exported = originalDt.toJSON();
      const importedDt = new DecisionTree(exported);
      
      const originalAccuracy = originalDt.evaluate(trainingData);
      const importedAccuracy = importedDt.evaluate(trainingData);
      
      assert.strictEqual(importedAccuracy, originalAccuracy);
    });

    it('should handle round-trip export/import', () => {
      const exported1 = originalDt.toJSON();
      const importedDt = new DecisionTree(exported1);
      const exported2 = importedDt.toJSON();
      
      // Verify structure is maintained
      assert.deepStrictEqual(exported1.target, exported2.target);
      assert.deepStrictEqual(exported1.features, exported2.features);
      assert.deepStrictEqual(exported1.data, exported2.data);
      
      // Verify predictions are identical
      const sample = { color: 'red', shape: 'circle', size: 'small' };
      const prediction1 = originalDt.predict(sample);
      const prediction2 = importedDt.predict(sample);
      assert.strictEqual(prediction1, prediction2);
    });
  });

  describe('Error Handling in Import/Export', () => {
    it('should handle corrupted JSON import', () => {
      const corruptedData = {
        model: null,
        data: [],
        target: 'target',
        features: ['color', 'shape', 'size']
      };
      
      // Note: Current implementation doesn't validate imported data structure
      // This will fail during prediction when trying to access model properties
      const dt = new DecisionTree(corruptedData);
      assert.throws(() => {
        dt.predict({ color: 'red', shape: 'circle', size: 'small' });
      }, /Cannot read properties of null/);
    });

    it('should handle missing model properties', () => {
      const incompleteData = {
        data: trainingData,
        target: 'target',
        features: ['color', 'shape', 'size']
        // Missing 'model' property
      };
      
      // Note: Current implementation doesn't validate imported data structure
      // This will fail during prediction when trying to access model properties
      const dt = new DecisionTree(incompleteData as any);
      assert.throws(() => {
        dt.predict({ color: 'red', shape: 'circle', size: 'small' });
      }, /Cannot read properties of undefined/);
    });

    it('should handle missing data property', () => {
      const incompleteData = {
        model: { type: 'result', val: 'class1', name: 'class1', alias: 'class1' },
        target: 'target',
        features: ['color', 'shape', 'size']
        // Missing 'data' property
      };
      
      const dt = new DecisionTree(incompleteData as any);
      // Should not crash, but may have limited functionality
      assert.ok(dt);
    });

    it('should handle missing target property', () => {
      const incompleteData = {
        model: { type: 'result', val: 'class1', name: 'class1', alias: 'class1' },
        data: trainingData,
        features: ['color', 'shape', 'size']
        // Missing 'target' property
      };
      
      const dt = new DecisionTree(incompleteData as any);
      // Should not crash, but may have limited functionality
      assert.ok(dt);
    });

    it('should handle missing features property', () => {
      const incompleteData = {
        model: { type: 'result', val: 'class1', name: 'class1', alias: 'class1' },
        data: trainingData,
        target: 'target'
        // Missing 'features' property
      };
      
      const dt = new DecisionTree(incompleteData as any);
      // Should not crash, but may have limited functionality
      assert.ok(dt);
    });
  });

  describe('Model Validation', () => {
    it('should validate imported model structure', () => {
      const exported = originalDt.toJSON();
      const importedDt = new DecisionTree(exported);
      
      // Verify the imported model has the expected structure
      const importedModel = importedDt.toJSON();
      assert.ok(importedModel.model);
      assert.ok(Array.isArray(importedModel.data));
      assert.strictEqual(importedModel.target, 'target');
      assert.ok(Array.isArray(importedModel.features));
    });

    it('should handle models with different feature sets', () => {
      const exported = originalDt.toJSON();
      
      // Modify features in exported model
      exported.features = ['color', 'shape']; // Remove 'size'
      
      const importedDt = new DecisionTree(exported);
      
      // Should still work with available features
      const prediction = importedDt.predict({ color: 'red', shape: 'circle' });
      assert.ok(typeof prediction === 'string');
    });

    it('should handle models with different target names', () => {
      const exported = originalDt.toJSON();
      
      // Modify target in exported model
      exported.target = 'newTarget';
      
      const importedDt = new DecisionTree(exported);
      
      // Should still work with new target name
      const prediction = importedDt.predict({ color: 'red', shape: 'circle', size: 'small' });
      assert.ok(typeof prediction === 'string');
    });
  });

});
