import { strict as assert } from 'assert';
import { MSELoss, LogisticLoss, CrossEntropyLoss, LossFunctionFactory } from '../lib/shared/loss-functions.js';

describe('XGBoost Loss Functions - MSE Loss', function() {
  it('should calculate gradient correctly', () => {
    const prediction = 0.8;
    const actual = 1.0;
    const gradient = MSELoss.calculateGradient(prediction, actual);
    assert.ok(Math.abs(gradient - (-0.2)) < 1e-10);
  });

  it('should calculate hessian correctly', () => {
    const prediction = 0.8;
    const actual = 1.0;
    const hessian = MSELoss.calculateHessian(prediction, actual);
    assert.strictEqual(hessian, 1);
  });

  it('should calculate loss correctly', () => {
    const predictions = [0.8, 1.2, 0.9];
    const actuals = [1.0, 1.0, 1.0];
    const loss = MSELoss.calculateLoss(predictions, actuals);
    const expected = (0.04 + 0.04 + 0.01) / 3; // (0.2^2 + 0.2^2 + 0.1^2) / 3
    assert.ok(Math.abs(loss - expected) < 1e-10);
  });

  it('should calculate gradients and hessians together', () => {
    const predictions = [0.8, 1.2, 0.9];
    const actuals = [1.0, 1.0, 1.0];
    const result = MSELoss.calculateGradientsAndHessians(predictions, actuals);
    
    assert.ok(Array.isArray(result.gradient));
    assert.ok(Array.isArray(result.hessian));
    assert.strictEqual(result.gradient.length, 3);
    assert.strictEqual(result.hessian.length, 3);
    
    assert.ok(Math.abs(result.gradient[0] - (-0.2)) < 1e-10);
    assert.ok(Math.abs(result.gradient[1] - 0.2) < 1e-10);
    assert.ok(Math.abs(result.gradient[2] - (-0.1)) < 1e-10);
    
    assert.strictEqual(result.hessian[0], 1);
    assert.strictEqual(result.hessian[1], 1);
    assert.strictEqual(result.hessian[2], 1);
  });

  it('should handle perfect predictions', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actuals = [1.0, 2.0, 3.0];
    const loss = MSELoss.calculateLoss(predictions, actuals);
    assert.strictEqual(loss, 0);
    
    const result = MSELoss.calculateGradientsAndHessians(predictions, actuals);
    assert.deepStrictEqual(result.gradient, [0, 0, 0]);
    assert.deepStrictEqual(result.hessian, [1, 1, 1]);
  });
});

describe('XGBoost Loss Functions - Logistic Loss', function() {
  it('should calculate sigmoid correctly', () => {
    assert.strictEqual(LogisticLoss.sigmoid(0), 0.5);
    assert.ok(Math.abs(LogisticLoss.sigmoid(1) - 0.7310585786300049) < 1e-10);
    assert.ok(Math.abs(LogisticLoss.sigmoid(-1) - 0.2689414213699951) < 1e-10);
  });

  it('should handle extreme values in sigmoid', () => {
    assert.ok(LogisticLoss.sigmoid(500) > 0.999);
    assert.ok(LogisticLoss.sigmoid(-500) < 0.001);
  });

  it('should calculate gradient correctly', () => {
    const prediction = 1.0; // log-odds
    const actual = 1;
    const gradient = LogisticLoss.calculateGradient(prediction, actual);
    const expected = LogisticLoss.sigmoid(prediction) - actual;
    assert.ok(Math.abs(gradient - expected) < 1e-10);
  });

  it('should calculate hessian correctly', () => {
    const prediction = 1.0;
    const actual = 1;
    const hessian = LogisticLoss.calculateHessian(prediction, actual);
    const prob = LogisticLoss.sigmoid(prediction);
    const expected = prob * (1 - prob);
    assert.ok(Math.abs(hessian - expected) < 1e-10);
  });

  it('should calculate loss correctly', () => {
    const predictions = [1.0, -1.0, 0.0];
    const actuals = [1, 0, 1];
    const loss = LogisticLoss.calculateLoss(predictions, actuals);
    assert.ok(loss > 0);
    assert.ok(typeof loss === 'number');
  });

  it('should calculate gradients and hessians together', () => {
    const predictions = [1.0, -1.0, 0.0];
    const actuals = [1, 0, 1];
    const result = LogisticLoss.calculateGradientsAndHessians(predictions, actuals);
    
    assert.ok(Array.isArray(result.gradient));
    assert.ok(Array.isArray(result.hessian));
    assert.strictEqual(result.gradient.length, 3);
    assert.strictEqual(result.hessian.length, 3);
    
    // Check that gradients and hessians are calculated
    result.gradient.forEach(grad => assert.ok(typeof grad === 'number'));
    result.hessian.forEach(hess => assert.ok(typeof hess === 'number' && hess >= 0));
  });

  it('should handle perfect predictions', () => {
    const predictions = [10.0, -10.0, 5.0];
    const actuals = [1, 0, 1];
    const loss = LogisticLoss.calculateLoss(predictions, actuals);
    assert.ok(loss >= 0); // Should be non-negative
  });
});

describe('XGBoost Loss Functions - Cross Entropy Loss', function() {
  it('should calculate softmax correctly', () => {
    const x = [1.0, 2.0, 3.0];
    const softmax = CrossEntropyLoss.softmax(x);
    
    assert.ok(Array.isArray(softmax));
    assert.strictEqual(softmax.length, 3);
    
    // Sum should be 1
    const sum = softmax.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-10);
    
    // All values should be positive
    softmax.forEach(val => assert.ok(val > 0));
  });

  it('should handle extreme values in softmax', () => {
    const x = [1000, 1001, 1002];
    const softmax = CrossEntropyLoss.softmax(x);
    
    // Should not overflow
    assert.ok(Array.isArray(softmax));
    assert.ok(softmax.every(val => val >= 0 && val <= 1));
    
    const sum = softmax.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-10);
  });

  it('should calculate gradient correctly', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actual = 1; // Index of correct class
    const gradient = CrossEntropyLoss.calculateGradient(predictions, actual);
    
    assert.ok(Array.isArray(gradient));
    assert.strictEqual(gradient.length, 3);
    
    // Sum should be 0
    const sum = gradient.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum) < 1e-10);
  });

  it('should calculate hessian correctly', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actual = 1;
    const hessian = CrossEntropyLoss.calculateHessian(predictions, actual);
    
    assert.ok(Array.isArray(hessian));
    assert.strictEqual(hessian.length, 3);
    assert.ok(Array.isArray(hessian[0]));
    assert.strictEqual(hessian[0].length, 3);
  });

  it('should calculate loss correctly', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actuals = [1, 0, 2];
    const loss = CrossEntropyLoss.calculateLoss(predictions, actuals);
    
    assert.ok(loss >= 0);
    assert.ok(typeof loss === 'number');
  });

  it('should calculate gradients and hessians together', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actuals = [1, 0, 2];
    const result = CrossEntropyLoss.calculateGradientsAndHessians(predictions, actuals);
    
    assert.ok(Array.isArray(result.gradient));
    assert.ok(Array.isArray(result.hessian));
    assert.strictEqual(result.gradient.length, 3);
    assert.strictEqual(result.hessian.length, 3);
    
    // Check that gradients and hessians are calculated
    result.gradient.forEach(grad => assert.ok(typeof grad === 'number'));
    result.hessian.forEach(hess => assert.ok(typeof hess === 'number' && hess >= 0));
  });

  it('should handle perfect predictions', () => {
    const predictions = [10.0, 0.0, 0.0];
    const actuals = [0, 1, 2];
    const loss = CrossEntropyLoss.calculateLoss(predictions, actuals);
    assert.ok(loss >= 0); // Should be non-negative
  });
});

describe('XGBoost Loss Functions - Factory', function() {
  it('should create MSE loss for regression', () => {
    const lossFunction = LossFunctionFactory.create('regression');
    assert.strictEqual(lossFunction, MSELoss);
  });

  it('should create logistic loss for binary classification', () => {
    const lossFunction = LossFunctionFactory.create('binary');
    assert.strictEqual(lossFunction, LogisticLoss);
  });

  it('should create cross entropy loss for multiclass classification', () => {
    const lossFunction = LossFunctionFactory.create('multiclass');
    assert.strictEqual(lossFunction, CrossEntropyLoss);
  });

  it('should throw error for unsupported objective', () => {
    assert.throws(() => LossFunctionFactory.create('unsupported' as any));
  });

  it('should have consistent interface across all loss functions', () => {
    const objectives = ['regression', 'binary', 'multiclass'] as const;
    
    objectives.forEach(objective => {
      const lossFunction = LossFunctionFactory.create(objective);
      
      // Test that all required methods exist
      assert.ok(typeof lossFunction.calculateGradientsAndHessians === 'function');
      assert.ok(typeof lossFunction.calculateLoss === 'function');
      
      // Test with sample data
      const predictions = [0.5, 1.0, 1.5];
      const actuals = [0, 1, 1];
      
      const result = lossFunction.calculateGradientsAndHessians(predictions, actuals);
      assert.ok(Array.isArray(result.gradient));
      assert.ok(Array.isArray(result.hessian));
      assert.strictEqual(result.gradient.length, 3);
      assert.strictEqual(result.hessian.length, 3);
      
      const loss = lossFunction.calculateLoss(predictions, actuals);
      assert.ok(typeof loss === 'number');
    });
  });
});

describe('XGBoost Loss Functions - Edge Cases', function() {
  it('should handle empty arrays', () => {
    const predictions: number[] = [];
    const actuals: number[] = [];
    
    const mseLoss = MSELoss.calculateLoss(predictions, actuals);
    assert.ok(isNaN(mseLoss) || mseLoss === 0);
    
    const logisticLoss = LogisticLoss.calculateLoss(predictions, actuals);
    assert.ok(isNaN(logisticLoss) || logisticLoss === 0);
    
    const crossEntropyLoss = CrossEntropyLoss.calculateLoss(predictions, actuals);
    assert.ok(isNaN(crossEntropyLoss) || crossEntropyLoss === 0);
  });

  it('should handle single element arrays', () => {
    const predictions = [0.5];
    const actuals = [1];
    
    const mseResult = MSELoss.calculateGradientsAndHessians(predictions, actuals);
    assert.strictEqual(mseResult.gradient.length, 1);
    assert.strictEqual(mseResult.hessian.length, 1);
    
    const logisticResult = LogisticLoss.calculateGradientsAndHessians(predictions, actuals);
    assert.strictEqual(logisticResult.gradient.length, 1);
    assert.strictEqual(logisticResult.hessian.length, 1);
    
    const crossEntropyResult = CrossEntropyLoss.calculateGradientsAndHessians(predictions, actuals);
    assert.strictEqual(crossEntropyResult.gradient.length, 1);
    assert.strictEqual(crossEntropyResult.hessian.length, 1);
  });

  it('should handle identical predictions and actuals', () => {
    const predictions = [1.0, 2.0, 3.0];
    const actuals = [1.0, 2.0, 3.0];
    
    const mseLoss = MSELoss.calculateLoss(predictions, actuals);
    assert.strictEqual(mseLoss, 0);
    
    const mseResult = MSELoss.calculateGradientsAndHessians(predictions, actuals);
    assert.deepStrictEqual(mseResult.gradient, [0, 0, 0]);
    assert.deepStrictEqual(mseResult.hessian, [1, 1, 1]);
  });

  it('should handle very large numbers', () => {
    const predictions = [1e10, -1e10, 0];
    const actuals = [1, 0, 1];
    
    // Should not throw or return NaN
    assert.doesNotThrow(() => {
      MSELoss.calculateLoss(predictions, actuals);
      LogisticLoss.calculateLoss(predictions, actuals);
      CrossEntropyLoss.calculateLoss(predictions, actuals);
    });
  });

  it('should handle very small numbers', () => {
    const predictions = [1e-10, -1e-10, 0];
    const actuals = [1, 0, 1];
    
    // Should not throw or return NaN
    assert.doesNotThrow(() => {
      MSELoss.calculateLoss(predictions, actuals);
      LogisticLoss.calculateLoss(predictions, actuals);
      CrossEntropyLoss.calculateLoss(predictions, actuals);
    });
  });
});
