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

	it('should train on the dataset', function() {
		assert.ok(dt.model);
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
});
