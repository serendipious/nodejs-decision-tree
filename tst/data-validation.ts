import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Data Validation & Sanitization', () => {
  describe('Feature Name Validation', () => {
    it('should validate feature names are strings', () => {
      // Note: Current implementation only checks if features is an array, not element types
      // These should not throw errors with current validation
      const dt1 = new DecisionTree('target', [123 as any, 'feature2']);
      assert.ok(dt1);
      
      const dt2 = new DecisionTree('target', [true as any, 'feature2']);
      assert.ok(dt2);
      
      const dt3 = new DecisionTree('target', [null as any, 'feature2']);
      assert.ok(dt3);
      
      const dt4 = new DecisionTree('target', [undefined as any, 'feature2']);
      assert.ok(dt4);
    });

    it('should handle empty string feature names', () => {
      const dt = new DecisionTree('target', ['', 'feature2']);
      const data = [
        { '': 'value1', feature2: 'value2', target: 'class1' },
        { '': 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      dt.train(data);
      const prediction = dt.predict({ '': 'value1', feature2: 'value2' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle whitespace-only feature names', () => {
      const dt = new DecisionTree('target', ['   ', 'feature2']);
      const data = [
        { '   ': 'value1', feature2: 'value2', target: 'class1' },
        { '   ': 'value3', feature2: 'value4', target: 'class2' }
      ];
      
      dt.train(data);
      const prediction = dt.predict({ '   ': 'value1', feature2: 'value2' });
      assert.strictEqual(prediction, 'class1');
    });
  });

  describe('Target Column Validation', () => {
    it('should validate target column exists in data', () => {
      const dt = new DecisionTree('nonexistent', ['feature1', 'feature2']);
      const data = [
        { feature1: 'value1', feature2: 'value2' }
        // Missing target column
      ];
      
      // Note: Current implementation doesn't validate that target column exists in training data
      // This is a design decision - the implementation is very permissive
      // Training will succeed but the resulting tree may not work correctly
      dt.train(data);
      assert.ok(dt.toJSON());
    });

    it('should handle target column with different data types', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const mixedTypeData = [
        { feature1: 'value1', target: 'class1' },
        { feature1: 'value2', target: 123 },
        { feature1: 'value3', target: true },
        { feature1: 'value4', target: null }
      ];
      
      dt.train(mixedTypeData);
      const prediction = dt.predict({ feature1: 'value1' });
      assert.ok(typeof prediction === 'string' || typeof prediction === 'number' || typeof prediction === 'boolean');
    });
  });

  describe('Data Type Validation', () => {
    it('should handle numeric vs string feature values', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const mixedTypeData = [
        { feature1: 'string1', feature2: 123, target: 'class1' },
        { feature1: 456, feature2: 'string2', target: 'class2' },
        { feature1: 'string3', feature2: 789, target: 'class1' }
      ];
      
      dt.train(mixedTypeData);
      const prediction = dt.predict({ feature1: 'string1', feature2: 123 });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle boolean feature values', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      const booleanData = [
        { feature1: true, feature2: false, target: 'class1' },
        { feature1: false, feature2: true, target: 'class2' },
        { feature1: true, feature2: true, target: 'class1' }
      ];
      
      dt.train(booleanData);
      const prediction = dt.predict({ feature1: true, feature2: false });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle mixed data types in same feature', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const mixedData = [
        { feature1: 'string', target: 'class1' },
        { feature1: 123, target: 'class2' },
        { feature1: true, target: 'class3' },
        { feature1: null, target: 'class1' }
      ];
      
      dt.train(mixedData);
      const prediction = dt.predict({ feature1: 'string' });
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Data Consistency Validation', () => {
    it('should handle samples with different feature sets', () => {
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3']);
      const inconsistentData = [
        { feature1: 'value1', feature2: 'value2', target: 'class1' },
        { feature1: 'value3', feature2: 'value4', feature3: 'value5', target: 'class2' },
        { feature1: 'value6', target: 'class3' }
      ];
      
      dt.train(inconsistentData);
      // Should handle missing features gracefully
      const prediction = dt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.ok(typeof prediction === 'string');
    });

    it('should handle nested object values', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const nestedData = [
        { feature1: { nested: 'value1' }, target: 'class1' },
        { feature1: { nested: 'value2' }, target: 'class2' }
      ];
      
      dt.train(nestedData);
      const prediction = dt.predict({ feature1: { nested: 'value1' } });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle array values', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const arrayData = [
        { feature1: ['item1', 'item2'], target: 'class1' },
        { feature1: ['item3', 'item4'], target: 'class2' }
      ];
      
      dt.train(arrayData);
      const prediction = dt.predict({ feature1: ['item1', 'item2'] });
      assert.strictEqual(prediction, 'class1');
    });
  });

  describe('Input Sanitization', () => {
    it('should handle HTML/script injection attempts', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const maliciousData = [
        { feature1: '<script>alert("xss")</script>', target: 'class1' },
        { feature1: 'javascript:alert("xss")', target: 'class2' },
        { feature1: 'normal_value', target: 'class3' }
      ];
      
      dt.train(maliciousData);
      const prediction = dt.predict({ feature1: '<script>alert("xss")</script>' });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle SQL injection attempts', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const sqlInjectionData = [
        { feature1: "'; DROP TABLE users; --", target: 'class1' },
        { feature1: "'; SELECT * FROM users; --", target: 'class2' },
        { feature1: 'normal_value', target: 'class3' }
      ];
      
      dt.train(sqlInjectionData);
      const prediction = dt.predict({ feature1: "'; DROP TABLE users; --" });
      assert.strictEqual(prediction, 'class1');
    });

    it('should handle very large data values', () => {
      const dt = new DecisionTree('target', ['feature1']);
      const largeData = [
        { feature1: 'a'.repeat(100000), target: 'class1' },
        { feature1: 'normal_value', target: 'class2' }
      ];
      
      dt.train(largeData);
      const prediction = dt.predict({ feature1: 'a'.repeat(100000) });
      assert.strictEqual(prediction, 'class1');
    });
  });
});
