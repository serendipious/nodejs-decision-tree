const assert = require('assert');
const ID3    = require('../lib/decision-tree');

const OBJECT_EVALUATION_DATASET = require('data/object-evaluation.json');
const TIC_TAC_TOE_DATASET       = require('data/tic-tac-toe.json');
const VOTING_DATASET            = require('data/voting.json');

describe('ID3 Decision Tree on Sample Datasets', function() {
  describe('Tic Tac Toe Dataset', function() {
    const dt = new ID3(TIC_TAC_TOE_DATASET.data, 'classification', TIC_TAC_TOE_DATASET.features);

    it('should initialize on training dataset', function() {
      assert.ok(dt);
      assert.ok(dt.toJSON());
    });

    it('should evaluate perfectly on training dataset', function() {
      const accuracy = dt.evaluate(TIC_TAC_TOE_DATASET.data);
      assert.strictEqual(accuracy, 1);
    });
  });

  describe('Voting Dataset', function() {
    const dt = new ID3(VOTING_DATASET.data, 'classification', VOTING_DATASET.features);
    
    it('should initialize on training dataset', function() {
      assert.ok(dt);
      assert.ok(dt.toJSON());
    });

    it('should evaluate perfectly on training dataset', function() {
      const accuracy = dt.evaluate(VOTING_DATASET.data);
      assert.strictEqual(accuracy, 1);
    });
  });

  describe('Object Evaluation Dataset', function() {
    const dt = new ID3(OBJECT_EVALUATION_DATASET.data, 'classification', OBJECT_EVALUATION_DATASET.features);

    it('should initialize on training dataset', function() {
      assert.ok(dt);
      assert.ok(dt.toJSON());
    });

    it('should evaluate perfectly on training dataset', function() {
      const data =  [
        {"foo":true, "bar":true, "flim":true, "classification":{"description":"foo bar flim"}},
        {"foo":false, "bar":true, "flim":true, "classification":{"description":"bar flim"}},
        {"foo":true, "bar":false, "flim":true, "classification":{"description":"foo flim"}},
        {"foo":false, "bar":false, "flim":true, "classification":{"description":"flim"}},
        {"foo":true, "bar":true, "flim":false, "classification":{"description":"foo bar"}},
        {"foo":false, "bar":true, "flim":false, "classification":{"description":"bar"}},
        {"foo":true, "bar":false, "flim":false, "classification":{"description":"foo"}},
        {"foo":false, "bar":false, "flim":false, "classification":{"description":"none"}}
      ];
      const accuracy = dt.evaluate(data);
      assert.strictEqual(accuracy, 1);
    });

    it('should evaluate 87.5% on training dataset', function() {
      const data =  [
        {"foo":true, "bar":true, "flim":true, "classification":{"description":"foo bar flim"}},
        {"foo":false, "bar":true, "flim":true, "classification":{"description":"bar flim"}},
        {"foo":true, "bar":false, "flim":true, "classification":{"description":"foo flim"}},
        {"foo":false, "bar":false, "flim":true, "classification":{"description":"flim"}},
        {"foo":true, "bar":true, "flim":false, "classification":{"description":"foo bar"}},
        {"foo":false, "bar":true, "flim":false, "classification":{"description":"bar"}},
        {"foo":true, "bar":false, "flim":false, "classification":{"description":"foo"}},
        {"foo":false, "bar":false, "flim":false, "classification":{}}
      ];
      const accuracy = dt.evaluate(data);
      assert.strictEqual(accuracy, 0.875);
    });
  });
});
