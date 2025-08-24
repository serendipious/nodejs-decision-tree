import { strict as assert } from 'assert';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import DecisionTree from '../lib/decision-tree.js';

// Type definitions for test data
interface SampleData {
  color: string;
  shape: string;
  liked: boolean;
}

interface Dataset {
  features: string[];
  data: SampleData[];
}

// Helper function to load JSON files
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadJSON<T>(filename: string): T {
  const filePath = join(__dirname, '..', 'data', filename);
  return JSON.parse(readFileSync(filePath, 'utf8')) as T;
}

const SAMPLE_DATASET = loadJSON<Dataset>('sample.json');
const SAMPLE_DATASET_CLASS_NAME = 'liked';

describe('Decision Tree Basics', function() {
  const dt = new DecisionTree(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);

  it('should initialize with valid argument constructor', () => {
    assert.ok(new DecisionTree(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
    assert.ok(new DecisionTree(SAMPLE_DATASET.data, SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
  });

  it('should initialize & train for the three argument constructor', function() {
    assert.ok(dt);
  });

  it('should throw initialization error with invalid constructor arguments', function() {
    assert.throws(() => new DecisionTree());
    assert.throws(() => new DecisionTree(1 as any, 2 as any, 3 as any, 4 as any));
    assert.throws(() => new DecisionTree(1 as any, 1 as any));
    assert.throws(() => new DecisionTree("abc", 1 as any));
    assert.throws(() => new DecisionTree(1 as any, 1 as any, 1 as any));
  });

  it('should train on the dataset', function() {
    dt.train(SAMPLE_DATASET.data);
    assert.ok(dt.toJSON());
  });

  it('should predict on a sample instance', function() {
    const sample = SAMPLE_DATASET.data[0];
    const predicted_class = dt.predict(sample);
    const actual_class = sample[SAMPLE_DATASET_CLASS_NAME as keyof SampleData];
    assert.strictEqual(predicted_class, actual_class);
  });

  it('should evaluate perfectly on training dataset', function() {
    const accuracy = dt.evaluate(SAMPLE_DATASET.data);
    assert.strictEqual(accuracy, 1);
  });

  it('should provide access to the underlying model as JSON', function() {
    const dtJson = dt.toJSON();
    const treeModel = dtJson.model;

    assert.strictEqual(treeModel.constructor, Object);
    assert.ok(Array.isArray(treeModel.vals));
    assert.strictEqual(treeModel.vals.length, 3);

    assert.ok(Array.isArray(dtJson.features));
    assert.strictEqual(typeof dtJson.target, 'string');
  });

  it('should provide access to insights on each node (e.g. gain, sample size, etc.)', () => {
    const dtJson = dt.toJSON();
    const rootNode = dtJson.model;

    assert.strictEqual(rootNode.gain! >= 0 && rootNode.gain! <= 1, true);
    assert.strictEqual(typeof rootNode.sampleSize, 'number');

    const childNodes = rootNode.vals!;
    for (let childNode of childNodes) {
      assert.strictEqual(typeof childNode.prob, 'number');
      assert.strictEqual(typeof childNode.sampleSize, 'number');
    }
  });

  it('should initialize from existing or previously exported model', function() {
    const pretrainedDecTree = new DecisionTree(dt.toJSON());
    const pretrainedDecTreeAccuracy = pretrainedDecTree.evaluate(SAMPLE_DATASET.data);
    assert.strictEqual(pretrainedDecTreeAccuracy, 1);
  });
});
