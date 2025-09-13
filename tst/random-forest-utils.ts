import { strict as assert } from 'assert';
import { 
  SeededRandom, 
  bootstrapSample, 
  selectRandomFeatures, 
  majorityVote 
} from '../lib/shared/utils.js';

describe('Random Forest Utility Functions', function() {
  describe('SeededRandom', function() {
    it('should generate consistent random numbers with same seed', () => {
      const random1 = new SeededRandom(42);
      const random2 = new SeededRandom(42);
      
      for (let i = 0; i < 10; i++) {
        assert.strictEqual(random1.next(), random2.next());
      }
    });

    it('should generate different random numbers with different seeds', () => {
      const random1 = new SeededRandom(42);
      const random2 = new SeededRandom(43);
      
      const values1 = [];
      const values2 = [];
      
      for (let i = 0; i < 10; i++) {
        values1.push(random1.next());
        values2.push(random2.next());
      }
      
      assert.notDeepStrictEqual(values1, values2);
    });

    it('should generate numbers between 0 and 1', () => {
      const random = new SeededRandom(42);
      
      for (let i = 0; i < 100; i++) {
        const value = random.next();
        assert.ok(value >= 0 && value < 1);
      }
    });

    it('should generate integers within range', () => {
      const random = new SeededRandom(42);
      
      for (let i = 0; i < 100; i++) {
        const value = random.nextInt(10);
        assert.ok(value >= 0 && value < 10);
        assert.ok(Number.isInteger(value));
      }
    });
  });

  describe('bootstrapSample', function() {
    it('should create bootstrap sample of correct size', () => {
      const data = [1, 2, 3, 4, 5];
      const random = new SeededRandom(42);
      const sample = bootstrapSample(data, 3, random);
      
      assert.strictEqual(sample.length, 3);
    });

    it('should create bootstrap sample with replacement', () => {
      const data = [1, 2, 3];
      const random = new SeededRandom(42);
      const sample = bootstrapSample(data, 10, random);
      
      assert.strictEqual(sample.length, 10);
      // With replacement, we can have more samples than original data
      sample.forEach(item => {
        assert.ok(data.includes(item));
      });
    });

    it('should be reproducible with same seed', () => {
      const data = [1, 2, 3, 4, 5];
      const random1 = new SeededRandom(42);
      const random2 = new SeededRandom(42);
      
      const sample1 = bootstrapSample(data, 5, random1);
      const sample2 = bootstrapSample(data, 5, random2);
      
      assert.deepStrictEqual(sample1, sample2);
    });

    it('should handle empty data', () => {
      const data: any[] = [];
      const random = new SeededRandom(42);
      
      assert.throws(() => bootstrapSample(data, 1, random));
    });

    it('should handle zero sample size', () => {
      const data = [1, 2, 3];
      const random = new SeededRandom(42);
      const sample = bootstrapSample(data, 0, random);
      
      assert.strictEqual(sample.length, 0);
    });
  });

  describe('selectRandomFeatures', function() {
    it('should select correct number of features for sqrt', () => {
      const features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 'sqrt', random);
      
      const expectedCount = Math.floor(Math.sqrt(features.length));
      assert.strictEqual(selected.length, expectedCount);
    });

    it('should select correct number of features for log2', () => {
      const features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 'log2', random);
      
      const expectedCount = Math.floor(Math.log2(features.length));
      assert.strictEqual(selected.length, expectedCount);
    });

    it('should select correct number of features for auto', () => {
      const features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 'auto', random);
      
      const expectedCount = Math.floor(Math.sqrt(features.length));
      assert.strictEqual(selected.length, expectedCount);
    });

    it('should select correct number of features for numeric', () => {
      const features = ['a', 'b', 'c', 'd', 'e'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 3, random);
      
      assert.strictEqual(selected.length, 3);
    });

    it('should select at least 1 feature even with very small maxFeatures', () => {
      const features = ['a', 'b', 'c', 'd', 'e'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 0, random);
      
      assert.strictEqual(selected.length, 1);
    });

    it('should not select more features than available', () => {
      const features = ['a', 'b', 'c'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 10, random);
      
      assert.strictEqual(selected.length, 3);
    });

    it('should select different features on different calls', () => {
      const features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
      const random1 = new SeededRandom(42);
      const random2 = new SeededRandom(43);
      
      const selected1 = selectRandomFeatures(features, 4, random1);
      const selected2 = selectRandomFeatures(features, 4, random2);
      
      assert.notDeepStrictEqual(selected1, selected2);
    });

    it('should be reproducible with same seed', () => {
      const features = ['a', 'b', 'c', 'd', 'e'];
      const random1 = new SeededRandom(42);
      const random2 = new SeededRandom(42);
      
      const selected1 = selectRandomFeatures(features, 3, random1);
      const selected2 = selectRandomFeatures(features, 3, random2);
      
      assert.deepStrictEqual(selected1, selected2);
    });

    it('should only select from original features', () => {
      const features = ['a', 'b', 'c', 'd', 'e'];
      const random = new SeededRandom(42);
      const selected = selectRandomFeatures(features, 3, random);
      
      selected.forEach(feature => {
        assert.ok(features.includes(feature));
      });
    });
  });

  describe('majorityVote', function() {
    it('should return majority vote for clear majority', () => {
      const predictions = [true, true, true, false, false];
      const result = majorityVote(predictions);
      assert.strictEqual(result, true);
    });

    it('should return majority vote for clear majority with different types', () => {
      const predictions = ['a', 'a', 'a', 'b', 'b'];
      const result = majorityVote(predictions);
      assert.strictEqual(result, 'a');
    });

    it('should handle tie-breaking', () => {
      const predictions = [true, true, false, false];
      const result = majorityVote(predictions);
      assert.ok(typeof result === 'boolean');
    });

    it('should handle single prediction', () => {
      const predictions = [true];
      const result = majorityVote(predictions);
      assert.strictEqual(result, true);
    });

    it('should handle empty predictions', () => {
      const predictions: any[] = [];
      const result = majorityVote(predictions);
      assert.strictEqual(result, null);
    });

    it('should handle all same predictions', () => {
      const predictions = [false, false, false, false];
      const result = majorityVote(predictions);
      assert.strictEqual(result, false);
    });

    it('should handle numeric predictions', () => {
      const predictions = [1, 1, 2, 2, 1];
      const result = majorityVote(predictions);
      assert.strictEqual(result, 1);
    });

    it('should handle mixed type predictions', () => {
      const predictions = ['1', '1', '2', '2', '1'];
      const result = majorityVote(predictions);
      assert.strictEqual(result, '1');
    });

    it('should handle boolean string predictions', () => {
      const predictions = ['true', 'true', 'false', 'false', 'true'];
      const result = majorityVote(predictions);
      assert.strictEqual(result, 'true');
    });

    it('should handle complex object predictions', () => {
      const predictions = [
        { value: 1 },
        { value: 1 },
        { value: 2 },
        { value: 2 },
        { value: 1 }
      ];
      const result = majorityVote(predictions);
      assert.strictEqual(result, '[object Object]');
    });
  });
});
