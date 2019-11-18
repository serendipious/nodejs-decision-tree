const SAMPLE_DATASET = require('data/sample.json');
const SAMPLE_DATASET_CLASS_NAME = 'liked';

var assert = require('assert');
var ID3 = require('lib/decision-tree');

describe('ID3 Decision Tree', function() {
  var dt;
  before(function() {
    dt = new ID3(SAMPLE_DATASET.data, SAMPLE_DATASET_CLASS_NAME, SAMPLE_DATASET.features);
  });

  it('should initialize', function() {
    assert.ok(dt);
  });

  it('should throw initialization error with invalid constructor arguments', function() {
    assert.throws(() => new ID3());
    assert.throws(() => new ID3(1, 2));
  });

  it('should train on the dataset', function() {
    assert.ok(dt.toJSON());
  });

  it('should predict on a sample instance', function() {
    var sample = SAMPLE_DATASET.data[0];
    var predicted_class = dt.predict(sample);
    var actual_class = sample[SAMPLE_DATASET_CLASS_NAME];
    assert.equal(predicted_class, actual_class);
  });

  it('should evaluate perfectly on training dataset', function() {
    var accuracy = dt.evaluate(SAMPLE_DATASET.data);
    assert.equal(accuracy, 1);
  });

  it('should provide access to the underlying model as JSON', function() {
    var dtJson = dt.toJSON();
    var treeModel = dtJson.model;
    assert.equal(treeModel.constructor, Object);
    assert.equal(treeModel.vals.constructor, Array);
    assert.equal(treeModel.vals.length, 3);

    assert.equal(dtJson.features.constructor, Array);
    assert.equal(dtJson.target.constructor, String);
  });

  it('should initialize from existing or previously exported model', function() {
    var pretrainedDecTree = new ID3(dt.toJSON());
    var pretrainedDecTreeAccuracy = pretrainedDecTree.evaluate(SAMPLE_DATASET.data);
    assert.equal(pretrainedDecTreeAccuracy, 1);
  });
});
