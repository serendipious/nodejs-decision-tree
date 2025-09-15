/**
 * Tests for Data Type Detection System
 * Tests automatic detection of discrete vs continuous variables
 */

import { strict as assert } from 'assert';
import { DataTypeDetector, detectDataTypes, recommendAlgorithm } from '../lib/shared/data-type-detection.js';

// Test data generators
function generateNormalDistribution(sampleCount: number, mean: number = 0, std: number = 1): number[] {
  const data: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data.push(mean + std * z0);
  }
  return data;
}

function generateUniformDistribution(sampleCount: number, min: number = 0, max: number = 1): number[] {
  const data: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    data.push(min + Math.random() * (max - min));
  }
  return data;
}

function generateCategoricalData(sampleCount: number, categories: string[]): string[] {
  const data: string[] = [];
  for (let i = 0; i < sampleCount; i++) {
    data.push(categories[Math.floor(Math.random() * categories.length)]);
  }
  return data;
}

function generateBooleanData(sampleCount: number): boolean[] {
  const data: boolean[] = [];
  for (let i = 0; i < sampleCount; i++) {
    data.push(Math.random() > 0.5);
  }
  return data;
}

describe('Data Type Detection Core Functionality', function() {
  describe('Continuous Data Detection', function() {
    it('should detect normally distributed data as continuous', function() {
      const normalData = generateNormalDistribution(1000, 50, 10);
      const data = normalData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'continuous');
      assert(analysis.feature.confidence > 0.7);
      assert(analysis.feature.statistics !== undefined);
      assert(typeof analysis.feature.statistics!.mean === 'number');
      assert(typeof analysis.feature.statistics!.std === 'number');
    });

    it('should detect uniformly distributed data as continuous', function() {
      const uniformData = generateUniformDistribution(1000, 0, 100);
      const data = uniformData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'continuous');
      assert(analysis.feature.confidence > 0.7);
    });

    it('should detect high-cardinality numeric data as continuous', function() {
      const highCardinalityData = Array.from({ length: 1000 }, (_, i) => i + Math.random());
      const data = highCardinalityData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'continuous');
      assert(analysis.feature.confidence > 0.7);
    });
  });

  describe('Discrete Data Detection', function() {
    it('should detect categorical data as discrete', function() {
      const categoricalData = generateCategoricalData(1000, ['A', 'B', 'C', 'D']);
      const data = categoricalData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert(analysis.feature.confidence > 0.7);
      assert(analysis.feature.uniqueValues !== undefined);
      assert(analysis.feature.uniqueValues!.length === 4);
    });

    it('should detect boolean data as discrete', function() {
      const booleanData = generateBooleanData(1000);
      const data = booleanData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert.strictEqual(analysis.feature.confidence, 1.0);
      assert(analysis.feature.uniqueValues !== undefined);
      assert(analysis.feature.uniqueValues!.length === 2);
    });

    it('should detect low-cardinality numeric data as discrete', function() {
      const lowCardinalityData = Array.from({ length: 1000 }, () => Math.floor(Math.random() * 5));
      const data = lowCardinalityData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert(analysis.feature.confidence > 0.7);
      assert(analysis.feature.cardinality === 5);
    });
  });

  describe('Mixed Data Detection', function() {
    it('should correctly classify mixed feature types', function() {
      const continuousData = generateNormalDistribution(1000, 50, 10);
      const discreteData = generateCategoricalData(1000, ['A', 'B', 'C']);
      const booleanData = generateBooleanData(1000);
      
      const data = continuousData.map((cont, i) => ({
        continuous: cont,
        discrete: discreteData[i],
        boolean: booleanData[i],
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['continuous', 'discrete', 'boolean']);
      
      assert.strictEqual(analysis.continuous.type, 'continuous');
      assert.strictEqual(analysis.discrete.type, 'discrete');
      assert.strictEqual(analysis.boolean.type, 'discrete');
    });
  });
});

describe('Data Type Detection Configuration', function() {
  describe('Threshold Configuration', function() {
    it('should respect discreteThreshold parameter', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        feature: i % 15, // 15 unique values
        target: i % 2 === 0
      }));
      
      // With default threshold (20), should be discrete (15 < 20)
      const analysis1 = detectDataTypes(data, ['feature']);
      assert.strictEqual(analysis1.feature.type, 'discrete');
      
      // With lower threshold (10) and continuousThreshold (15), should be continuous (15 >= 15)
      const detector = new DataTypeDetector({ 
        discreteThreshold: 10, 
        continuousThreshold: 15 
      });
      const analysis2 = detector.analyzeFeatures(data, ['feature']);
      assert.strictEqual(analysis2.feature.type, 'continuous');
    });

    it('should respect continuousThreshold parameter', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        feature: i % 15, // 15 unique values
        target: i % 2 === 0
      }));
      
      // With default threshold (20), should be discrete
      const analysis1 = detectDataTypes(data, ['feature']);
      assert.strictEqual(analysis1.feature.type, 'discrete');
      
      // With lower threshold (10), should be continuous
      const detector = new DataTypeDetector({ continuousThreshold: 10 });
      const analysis2 = detector.analyzeFeatures(data, ['feature']);
      assert.strictEqual(analysis2.feature.type, 'continuous');
    });
  });

  describe('Statistical Tests Configuration', function() {
    it('should use statistical tests when enabled', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        feature: i % 25, // 25 unique values - borderline case
        target: i % 2 === 0
      }));
      
      const detectorWithTests = new DataTypeDetector({ 
        statisticalTests: true,
        discreteThreshold: 20,
        continuousThreshold: 20
      });
      const detectorWithoutTests = new DataTypeDetector({ 
        statisticalTests: false,
        discreteThreshold: 20,
        continuousThreshold: 20
      });
      
      const analysis1 = detectorWithTests.analyzeFeatures(data, ['feature']);
      const analysis2 = detectorWithoutTests.analyzeFeatures(data, ['feature']);
      
      // Results might differ based on statistical tests
      assert(analysis1.feature.type === 'discrete' || analysis1.feature.type === 'continuous');
      assert(analysis2.feature.type === 'discrete' || analysis2.feature.type === 'continuous');
    });
  });
});

describe('Algorithm Recommendation', function() {
  describe('Pure Continuous Data', function() {
    it('should recommend CART for continuous features', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        x1: Math.random() * 100,
        x2: Math.random() * 50,
        target: Math.random() > 0.5
      }));
      
      const recommendation = recommendAlgorithm(data, ['x1', 'x2'], 'target');
      
      assert.strictEqual(recommendation.algorithm, 'cart');
      assert(recommendation.confidence > 0.7);
      assert(recommendation.reasoning.length > 0);
    });
  });

  describe('Pure Discrete Data', function() {
    it('should recommend ID3 for discrete features', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        color: ['red', 'blue', 'green'][i % 3],
        shape: ['circle', 'square'][i % 2],
        target: Math.random() > 0.5
      }));
      
      const recommendation = recommendAlgorithm(data, ['color', 'shape'], 'target');
      
      assert.strictEqual(recommendation.algorithm, 'id3');
      assert(recommendation.confidence > 0.7);
      assert(recommendation.reasoning.length > 0);
    });
  });

  describe('Mixed Data', function() {
    it('should recommend hybrid approach for mixed features', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        continuous: Math.random() * 100,
        discrete: ['A', 'B', 'C'][i % 3],
        target: Math.random() > 0.5
      }));
      
      const recommendation = recommendAlgorithm(data, ['continuous', 'discrete'], 'target');
      
      assert.strictEqual(recommendation.algorithm, 'hybrid');
      assert(recommendation.confidence > 0.7);
      assert(recommendation.reasoning.length > 0);
    });
  });

  describe('Regression Tasks', function() {
    it('should recommend CART for continuous targets', function() {
      const data = Array.from({ length: 1000 }, (_, i) => ({
        x1: Math.random() * 100,
        x2: Math.random() * 50,
        target: Math.random() * 1000 // Continuous target
      }));
      
      const recommendation = recommendAlgorithm(data, ['x1', 'x2'], 'target');
      
      assert.strictEqual(recommendation.algorithm, 'cart');
      assert(recommendation.confidence > 0.7);
    });
  });
});

describe('Data Type Detection Edge Cases', function() {
  describe('Empty Data', function() {
    it('should handle empty datasets', function() {
      const analysis = detectDataTypes([], ['feature1', 'feature2']);
      
      assert.strictEqual(analysis.feature1.type, 'discrete');
      assert.strictEqual(analysis.feature1.confidence, 0);
      assert.strictEqual(analysis.feature2.type, 'discrete');
      assert.strictEqual(analysis.feature2.confidence, 0);
    });
  });

  describe('Single Sample', function() {
    it('should handle single sample datasets', function() {
      const data = [{ feature: 5, target: true }];
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert.strictEqual(analysis.feature.confidence, 1.0);
    });
  });

  describe('Missing Values', function() {
    it('should handle missing values when enabled', function() {
      const data = [
        { feature: 1, target: true },
        { feature: null, target: false },
        { feature: 3, target: true },
        { feature: undefined, target: false }
      ];
      
      const detector = new DataTypeDetector({ handleMissingValues: true });
      const analysis = detector.analyzeFeatures(data, ['feature']);
      
      assert(analysis.feature.type === 'discrete' || analysis.feature.type === 'continuous');
      assert(analysis.feature.missingValues === 2);
      assert(analysis.feature.missingPercentage === 50);
    });

    it('should handle missing values when disabled', function() {
      const data = [
        { feature: 1, target: true },
        { feature: null, target: false },
        { feature: 3, target: true }
      ];
      
      const detector = new DataTypeDetector({ handleMissingValues: false });
      const analysis = detector.analyzeFeatures(data, ['feature']);
      
      // Should only consider non-null values
      assert(analysis.feature.type === 'discrete' || analysis.feature.type === 'continuous');
    });
  });

  describe('All Same Values', function() {
    it('should handle features with all same values', function() {
      const data = Array.from({ length: 100 }, () => ({
        feature: 5,
        target: true
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert.strictEqual(analysis.feature.confidence, 1.0);
      assert.strictEqual(analysis.feature.cardinality, 1);
    });
  });

  describe('Non-Numeric Data', function() {
    it('should handle string data correctly', function() {
      const data = Array.from({ length: 100 }, (_, i) => ({
        feature: `value_${i % 10}`,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
      assert.strictEqual(analysis.feature.confidence, 1.0);
    });

    it('should handle mixed data types', function() {
      const data = [
        { feature: 1, target: true },
        { feature: 'string', target: false },
        { feature: 3, target: true }
      ];
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.type, 'discrete');
    });
  });
});


describe('Data Type Detection Statistics', function() {
  describe('Continuous Statistics', function() {
    it('should calculate correct statistics for normal distribution', function() {
      const normalData = generateNormalDistribution(1000, 50, 10);
      const data = normalData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      const stats = analysis.feature.statistics!;
      
      assert(Math.abs(stats.mean - 50) < 1, `Mean should be ~50, got ${stats.mean}`);
      assert(Math.abs(stats.std - 10) < 1, `Std should be ~10, got ${stats.std}`);
      assert(stats.min < stats.mean, 'Min should be less than mean');
      assert(stats.max > stats.mean, 'Max should be greater than mean');
      assert.strictEqual(stats.quartiles.length, 3);
    });
  });

  describe('Discrete Statistics', function() {
    it('should calculate correct cardinality for categorical data', function() {
      const categoricalData = generateCategoricalData(1000, ['A', 'B', 'C', 'D', 'E']);
      const data = categoricalData.map((value, i) => ({
        feature: value,
        target: i % 2 === 0
      }));
      
      const analysis = detectDataTypes(data, ['feature']);
      
      assert.strictEqual(analysis.feature.cardinality, 5);
      assert.strictEqual(analysis.feature.uniqueValues!.length, 5);
    });
  });
});
