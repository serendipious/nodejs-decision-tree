import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Prediction Edge Cases', () => {
  let dt: DecisionTree;

  beforeEach(() => {
    // Setup a basic decision tree for testing
    const trainingData = [
      { color: 'red', shape: 'circle', size: 'small', target: 'class1' },
      { color: 'blue', shape: 'square', size: 'medium', target: 'class2' },
      { color: 'green', shape: 'triangle', size: 'large', target: 'class3' },
      { color: 'red', shape: 'square', size: 'medium', target: 'class1' },
      { color: 'blue', shape: 'circle', size: 'small', target: 'class2' }
    ];
    
    dt = new DecisionTree('target', ['color', 'shape', 'size']);
    dt.train(trainingData);
  });

  describe('Missing Features in Prediction', () => {
    it('should handle missing features in prediction sample', () => {
      const incompleteSample = { color: 'red', shape: 'circle' };
      // Missing 'size' feature
      
      const prediction = dt.predict(incompleteSample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });

    it('should handle completely empty prediction sample', () => {
      const emptySample = {};
      
      const prediction = dt.predict(emptySample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });

    it('should handle sample with only some features', () => {
      const partialSample = { color: 'red' };
      // Missing 'shape' and 'size' features
      
      const prediction = dt.predict(partialSample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });
  });

  describe('Unknown Feature Values', () => {
    it('should handle unknown feature values not seen during training', () => {
      const unknownSample = { 
        color: 'purple',  // Not in training data
        shape: 'hexagon', // Not in training data
        size: 'extra-large' // Not in training data
      };
      
      const prediction = dt.predict(unknownSample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });

    it('should handle mixed known and unknown values', () => {
      const mixedSample = { 
        color: 'red',     // Known value
        shape: 'hexagon', // Unknown value
        size: 'small'     // Known value
      };
      
      const prediction = dt.predict(mixedSample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });
  });

  describe('Extra Features in Prediction', () => {
    it('should handle extra features not used in training', () => {
      const extraFeatureSample = { 
        color: 'red', 
        shape: 'circle', 
        size: 'small',
        extraFeature: 'extra',      // Extra feature
        anotherFeature: 'another'   // Another extra feature
      };
      
      const prediction = dt.predict(extraFeatureSample);
      assert.ok(typeof prediction === 'string');
      assert.ok(['class1', 'class2', 'class3'].includes(prediction));
    });

    it('should ignore extra features and use only training features', () => {
      const extraFeatureSample = { 
        color: 'red', 
        shape: 'circle', 
        size: 'small',
        unusedFeature: 'unused'
      };
      
      const prediction = dt.predict(extraFeatureSample);
      // Should behave the same as { color: 'red', shape: 'circle', size: 'small' }
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Data Type Mismatches', () => {
    it('should handle numeric vs string feature values', () => {
      const numericSample = { 
        color: 123,        // Numeric instead of string
        shape: 'circle', 
        size: 'small'
      };
      
      const prediction = dt.predict(numericSample);
      assert.ok(typeof prediction === 'string');
    });

    it('should handle boolean feature values', () => {
      const booleanSample = { 
        color: true,       // Boolean instead of string
        shape: 'circle', 
        size: 'small'
      };
      
      const prediction = dt.predict(booleanSample);
      assert.ok(typeof prediction === 'string');
    });

    it('should handle null and undefined values in prediction', () => {
      const nullSample = { 
        color: null,       // Null value
        shape: 'circle', 
        size: 'small'
      };
      
      const undefinedSample = { 
        color: 'red', 
        shape: undefined,  // Undefined value
        size: 'small'
      };
      
      const nullPrediction = dt.predict(nullSample);
      const undefinedPrediction = dt.predict(undefinedSample);
      
      assert.ok(typeof nullPrediction === 'string');
      assert.ok(typeof undefinedPrediction === 'string');
    });
  });

  describe('Boundary Conditions', () => {
    it('should handle very long string values', () => {
      const longValueSample = { 
        color: 'a'.repeat(10000),  // Very long string
        shape: 'circle', 
        size: 'small'
      };
      
      const prediction = dt.predict(longValueSample);
      assert.ok(typeof prediction === 'string');
    });

    it('should handle special characters in feature values', () => {
      const specialCharSample = { 
        color: 'red!@#$%^&*()', 
        shape: 'circle', 
        size: 'small'
      };
      
      const prediction = dt.predict(specialCharSample);
      assert.ok(typeof prediction === 'string');
    });

    it('should handle unicode characters', () => {
      const unicodeSample = { 
        color: 'red🚀🎉', 
        shape: 'circle', 
        size: 'small'
      };
      
      const prediction = dt.predict(unicodeSample);
      assert.ok(typeof prediction === 'string');
    });
  });
});
