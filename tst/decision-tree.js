const SAMPLE_DATASET = require('data/sample.json');
const SAMPLE_DATASET_CLASS_NAME = 'liked';

var assert = require('assert');
var ID3 = require('../lib/decision-tree');

describe('ID3 Decision Tree Basics', function() {
  var dt = new ID3(SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);

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
  });

  it('should train on the dataset', function() {
    dt.train(SAMPLE_DATASET.data);
    assert.ok(dt.toJSON());
  });

  it('should predict on a sample instance', function() {
    var sample = SAMPLE_DATASET.data[0];
    var predicted_class = dt.predict(sample);
    var actual_class = sample[SAMPLE_DATASET_CLASS_NAME];
    assert.strictEqual(predicted_class, actual_class);
  });

  it('should evaluate perfectly on training dataset', function() {
    var accuracy = dt.evaluate(SAMPLE_DATASET.data);
    assert.strictEqual(accuracy, 1);
  });

  it('should provide access to the underlying model as JSON', function() {
    var dtJson = dt.toJSON();
    var treeModel = dtJson.model;
    assert.strictEqual(treeModel.constructor, Object);
    assert.strictEqual(treeModel.vals.constructor, Array);
    assert.strictEqual(treeModel.vals.length, 3);

    assert.strictEqual(dtJson.features.constructor, Array);
    assert.strictEqual(dtJson.target.constructor, String);
  });

  it('should initialize from existing or previously exported model', function() {
    var pretrainedDecTree = new ID3(dt.toJSON());
    var pretrainedDecTreeAccuracy = pretrainedDecTree.evaluate(SAMPLE_DATASET.data);
    assert.strictEqual(pretrainedDecTreeAccuracy, 1);
  });
});
