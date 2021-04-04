const assert = require('assert');
const ID3 = require('../lib/decision-tree');

const SAMPLE_DATASET = require('data/sample.json');
const SAMPLE_DATASET_CLASS_NAME = 'liked';

describe('ID3 Decision Tree Basics', function() {
  const dt = new ID3(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);

  it('should initialize with valid argument constructor', () => {
    assert.ok(new ID3(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
    assert.ok(new ID3(SAMPLE_DATASET.data, SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features));
  });

  it('should initialize & train for the three argument constructor', function() {
    assert.ok(dt);
  });

  it('should throw initialization error with invalid constructor arguments', function() {
    assert.throws(() => new ID3());
    assert.throws(() => new ID3(1, 2, 3, 4));
    assert.throws(() => new ID3(1, 1));
    assert.throws(() => new ID3("abc", 1));
    assert.throws(() => new ID3(1, 1, 1));
  });

  it('should train on the dataset', function() {
    dt.train(SAMPLE_DATASET.data);
    assert.ok(dt.toJSON());
  });

  it('should predict on a sample instance', function() {
    const sample = SAMPLE_DATASET.data[0];
    const predicted_class = dt.predict(sample);
    const actual_class = sample[SAMPLE_DATASET_CLASS_NAME];
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
    assert.strictEqual(treeModel.vals.constructor, Array);
    assert.strictEqual(treeModel.vals.length, 3);

    assert.strictEqual(dtJson.features.constructor, Array);
    assert.strictEqual(dtJson.target.constructor, String);
  });

  it('should provide access to insights on each node (e.g. gain, sample size, etc.)', () => {
    const dtJson = dt.toJSON();
    const rootNode = dtJson.model;

    assert.strictEqual(rootNode.gain >= 0 && rootNode.gain <= 1, true);
    assert.strictEqual(rootNode.sampleSize.constructor, Number);

    const childNodes = rootNode.vals;
    for (let childNode of childNodes) {
      assert.strictEqual(childNode.prob.constructor, Number);
      assert.strictEqual(childNode.sampleSize.constructor, Number);
    }
  });

  it('should initialize from existing or previously exported model', function() {
    const pretrainedDecTree = new ID3(dt.toJSON());
    const pretrainedDecTreeAccuracy = pretrainedDecTree.evaluate(SAMPLE_DATASET.data);
    assert.strictEqual(pretrainedDecTreeAccuracy, 1);
  });
});
