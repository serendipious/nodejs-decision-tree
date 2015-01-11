const TIC_TAC_TOE_DATASET = require('data/tic-tac-toe.json');
const VOTING_DATASET = require('data/voting.json');

var assert = require('assert');
var ID3 = require('lib/decision-tree');

describe('ID3 Decision Tree', function() {
  describe('Tic Tac Toe Dataset', function() {
    var dt;
    before(function() {
      dt = new ID3(TIC_TAC_TOE_DATASET.data, 'classification', TIC_TAC_TOE_DATASET.features);
    });

    it('should initialize on training dataset', function() {
      assert.ok(dt);
      assert.ok(dt.toJSON());
    });

    it('should evaluate perfectly on training dataset', function() {
      var accuracy = dt.evaluate(TIC_TAC_TOE_DATASET.data);
      assert.equal(accuracy, 1);
    });
  });

  describe('Voting Dataset', function() {
    var dt;
    before(function() {
      dt = new ID3(VOTING_DATASET.data, 'classification', VOTING_DATASET.features);
    });

    it('should initialize on training dataset', function() {
      assert.ok(dt);
      assert.ok(dt.toJSON());
    });

    it('should evaluate perfectly on training dataset', function() {
      var accuracy = dt.evaluate(VOTING_DATASET.data);
      assert.equal(accuracy, 1);
    });
  });
});
