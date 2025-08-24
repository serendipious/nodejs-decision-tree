import { strict as assert } from 'assert';
import DecisionTree from '../lib/decision-tree.js';

describe('Performance & Scalability Tests', () => {
  describe('Large Dataset Performance', () => {
    it('should handle large datasets efficiently (1000+ samples)', () => {
      const startTime = Date.now();
      
      // Create large dataset
      const largeData = [];
      for (let i = 0; i < 1000; i++) {
        largeData.push({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          feature3: `value${i % 3}`,
          feature4: `value${i % 7}`,
          feature5: `value${i % 4}`,
          target: `class${i % 3 + 1}`
        });
      }
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']);
      dt.train(largeData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time (adjust threshold as needed)
      assert.ok(trainingTime < 5000, `Training took too long: ${trainingTime}ms`);
      
      // Test prediction performance
      const predictStart = Date.now();
      for (let i = 0; i < 100; i++) {
        dt.predict({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          feature3: `value${i % 3}`,
          feature4: `value${i % 7}`,
          feature5: `value${i % 4}`
        });
      }
      const predictTime = Date.now() - predictStart;
      
      // Prediction should be fast
      assert.ok(predictTime < 1000, `Prediction took too long: ${predictTime}ms`);
    });

    it('should handle many features efficiently (50+ features)', () => {
      const startTime = Date.now();
      
      // Create dataset with many features
      const features = [];
      for (let i = 0; i < 50; i++) {
        features.push(`feature${i}`);
      }
      
      const manyFeaturesData = [];
      for (let i = 0; i < 100; i++) {
        const sample: any = { target: `class${i % 3 + 1}` };
        for (let j = 0; j < 50; j++) {
          sample[`feature${j}`] = `value${i % (j + 2)}`;
        }
        manyFeaturesData.push(sample);
      }
      
      const dt = new DecisionTree('target', features);
      dt.train(manyFeaturesData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time
      assert.ok(trainingTime < 3000, `Training with many features took too long: ${trainingTime}ms`);
      
      // Test prediction
      const sample: any = {};
      for (let j = 0; j < 50; j++) {
        sample[`feature${j}`] = `value0`;
      }
      
      const prediction = dt.predict(sample);
      assert.ok(typeof prediction === 'string');
    });

    it('should handle memory usage appropriately', () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Create large dataset
      const largeData = [];
      for (let i = 0; i < 5000; i++) {
        largeData.push({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          feature3: `value${i % 3}`,
          target: `class${i % 3 + 1}`
        });
      }
      
      const dt = new DecisionTree('target', ['feature1', 'feature2', 'feature3']);
      dt.train(largeData);
      
      const afterTrainingMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = afterTrainingMemory - initialMemory;
      
      // Memory increase should be reasonable (adjust threshold as needed)
      assert.ok(memoryIncrease < 100 * 1024 * 1024, `Memory increase too high: ${Math.round(memoryIncrease / 1024 / 1024)}MB`);
      
      // Clean up
      const exported = dt.toJSON();
      assert.ok(exported);
    });
  });

  describe('Deep Tree Performance', () => {
    it('should handle deep tree structures efficiently', () => {
      const startTime = Date.now();
      
      // Create very deep tree
      const deepData = [];
      for (let i = 0; i < 200; i++) {
        const binary = i.toString(2).padStart(8, '0');
        const sample: any = { target: `class${i % 3 + 1}` };
        for (let j = 0; j < 8; j++) {
          sample[`level${j + 1}`] = binary[j];
        }
        deepData.push(sample);
      }
      
      const features = ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8'];
      const dt = new DecisionTree('target', features);
      dt.train(deepData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time
      assert.ok(trainingTime < 5000, `Deep tree training took too long: ${trainingTime}ms`);
      
      // Test prediction performance
      const predictStart = Date.now();
      for (let i = 0; i < 50; i++) {
        const binary = i.toString(2).padStart(8, '0');
        const sample: any = {};
        for (let j = 0; j < 8; j++) {
          sample[`level${j + 1}`] = binary[j];
        }
        dt.predict(sample);
      }
      const predictTime = Date.now() - predictStart;
      
      // Prediction should be fast even for deep trees
      assert.ok(predictTime < 1000, `Deep tree prediction took too long: ${predictTime}ms`);
    });

    it('should handle wide tree structures efficiently', () => {
      const startTime = Date.now();
      
      // Create wide tree with many feature values
      const wideData = [];
      for (let i = 0; i < 100; i++) {
        const sample: any = { target: `class${i % 3 + 1}` };
        for (let j = 0; j < 5; j++) {
          sample[`feature${j}`] = `value${i % 20}`; // 20 different values per feature
        }
        wideData.push(sample);
      }
      
      const features = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4'];
      const dt = new DecisionTree('target', features);
      dt.train(wideData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time
      assert.ok(trainingTime < 3000, `Wide tree training took too long: ${trainingTime}ms`);
      
      // Test prediction
      const prediction = dt.predict({
        feature0: 'value1',
        feature1: 'value2',
        feature2: 'value3',
        feature3: 'value4',
        feature4: 'value5'
      });
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Real-World Scenarios', () => {
    it('should handle imbalanced datasets efficiently', () => {
      const startTime = Date.now();
      
      // Create imbalanced dataset (90/10 distribution)
      const imbalancedData = [];
      for (let i = 0; i < 900; i++) {
        imbalancedData.push({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          target: 'class1'
        });
      }
      for (let i = 0; i < 100; i++) {
        imbalancedData.push({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          target: 'class2'
        });
      }
      
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      dt.train(imbalancedData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time
      assert.ok(trainingTime < 3000, `Imbalanced dataset training took too long: ${trainingTime}ms`);
      
      // Test prediction accuracy
      const accuracy = dt.evaluate(imbalancedData);
      assert.ok(accuracy > 0.8, `Accuracy too low: ${accuracy}`);
    });

    it('should handle noisy data efficiently', () => {
      const startTime = Date.now();
      
      // Create noisy dataset with some contradictory samples
      const noisyData = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'A', feature2: 'X', target: 'class2' }, // Contradictory
        { feature1: 'B', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'Y', target: 'class2' },
        { feature1: 'B', feature2: 'Y', target: 'class1' }, // Contradictory
        { feature1: 'C', feature2: 'Z', target: 'class3' },
        { feature1: 'C', feature2: 'Z', target: 'class3' }
      ];
      
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      dt.train(noisyData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete quickly
      assert.ok(trainingTime < 1000, `Noisy data training took too long: ${trainingTime}ms`);
      
      // Test prediction
      const prediction = dt.predict({ feature1: 'A', feature2: 'X' });
      assert.ok(['class1', 'class2'].includes(prediction));
    });

    it('should handle incremental training scenarios', () => {
      const startTime = Date.now();
      
      // Initial training
      const initialData = [
        { feature1: 'A', feature2: 'X', target: 'class1' },
        { feature1: 'B', feature2: 'Y', target: 'class2' }
      ];
      
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      dt.train(initialData);
      
      // Additional training data
      const additionalData = [
        { feature1: 'A', feature2: 'Z', target: 'class1' },
        { feature1: 'B', feature2: 'W', target: 'class2' },
        { feature1: 'C', feature2: 'X', target: 'class3' }
      ];
      
      // Note: This tests the current behavior, but the current implementation
      // doesn't support incremental training - it retrains completely
      dt.train([...initialData, ...additionalData]);
      
      const totalTrainingTime = Date.now() - startTime;
      
      // Total training should complete within reasonable time
      assert.ok(totalTrainingTime < 2000, `Incremental training took too long: ${totalTrainingTime}ms`);
      
      // Test prediction with new data
      const prediction = dt.predict({ feature1: 'C', feature2: 'X' });
      assert.strictEqual(prediction, 'class3');
    });

    it('should handle feature scaling scenarios', () => {
      const startTime = Date.now();
      
      // Create dataset with features in different ranges
      const scaledData = [];
      for (let i = 0; i < 100; i++) {
        scaledData.push({
          smallFeature: i % 5,           // 0-4
          mediumFeature: i % 100,        // 0-99
          largeFeature: i % 10000,       // 0-9999
          categoricalFeature: `cat${i % 10}`, // categorical
          target: `class${i % 3 + 1}`
        });
      }
      
      const features = ['smallFeature', 'mediumFeature', 'largeFeature', 'categoricalFeature'];
      const dt = new DecisionTree('target', features);
      dt.train(scaledData);
      
      const trainingTime = Date.now() - startTime;
      
      // Training should complete within reasonable time
      assert.ok(trainingTime < 2000, `Scaled features training took too long: ${trainingTime}ms`);
      
      // Test prediction
      const prediction = dt.predict({
        smallFeature: 2,
        mediumFeature: 50,
        largeFeature: 5000,
        categoricalFeature: 'cat5'
      });
      assert.ok(typeof prediction === 'string');
    });
  });

  describe('Memory and Resource Management', () => {
    it('should handle memory cleanup after large operations', () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Perform large operation
      const largeData = [];
      for (let i = 0; i < 10000; i++) {
        largeData.push({
          feature1: `value${i % 10}`,
          feature2: `value${i % 5}`,
          target: `class${i % 3 + 1}`
        });
      }
      
      const dt = new DecisionTree('target', ['feature1', 'feature2']);
      dt.train(largeData);
      
      // Export and import to test memory handling
      const exported = dt.toJSON();
      const importedDt = new DecisionTree(exported);
      
      // Test prediction
      const prediction = importedDt.predict({ feature1: 'value1', feature2: 'value2' });
      assert.ok(typeof prediction === 'string');
      
      // Memory should not grow excessively
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Allow some memory increase but not excessive
      assert.ok(memoryIncrease < 200 * 1024 * 1024, `Memory increase too high: ${Math.round(memoryIncrease / 1024 / 1024)}MB`);
    });

    it('should handle concurrent operations efficiently', () => {
      const startTime = Date.now();
      
      // Create multiple decision trees concurrently
      const trees = [];
      for (let i = 0; i < 5; i++) {
        const data = [
          { feature1: `value${i}`, feature2: 'X', target: `class${i + 1}` },
          { feature1: `value${i}`, feature2: 'Y', target: `class${i + 1}` }
        ];
        
        const dt = new DecisionTree('target', ['feature1', 'feature2']);
        dt.train(data);
        trees.push(dt);
      }
      
      // Test concurrent predictions
      const predictions = trees.map((dt, i) => 
        dt.predict({ feature1: `value${i}`, feature2: 'X' })
      );
      
      const totalTime = Date.now() - startTime;
      
      // Concurrent operations should complete within reasonable time
      assert.ok(totalTime < 3000, `Concurrent operations took too long: ${totalTime}ms`);
      
      // Verify predictions
      for (let i = 0; i < predictions.length; i++) {
        assert.strictEqual(predictions[i], `class${i + 1}`);
      }
    });
  });
});
