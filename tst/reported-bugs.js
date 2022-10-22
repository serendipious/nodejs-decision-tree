const assert = require('assert');
const DecisionTree = require('../lib/decision-tree');

const OBJECT_EVALUATION_DATASET = require('data/object-evaluation.json');
const TIC_TAC_TOE_DATASET       = require('data/tic-tac-toe.json');
const VOTING_DATASET            = require('data/voting.json');

/**
 * Reported bugs from: https://github.com/serendipious/nodejs-decision-tree/issues
 */
describe('Decision Tree Reported Bugs', function() {
  /**
   * https://github.com/serendipious/nodejs-decision-tree/issues/21
   */
  it('should work with multiple decision tree instance declarations', () => {
    const dt1 = new DecisionTree(TIC_TAC_TOE_DATASET.data, 'classification', TIC_TAC_TOE_DATASET.features);
    const dt2 = new DecisionTree(VOTING_DATASET.data, 'classification', VOTING_DATASET.features);
    const dt3 = new DecisionTree(OBJECT_EVALUATION_DATASET.data, 'classification', OBJECT_EVALUATION_DATASET.features);

    assert.strictEqual(dt1.evaluate(TIC_TAC_TOE_DATASET.data), 1);
    assert.strictEqual(dt2.evaluate(VOTING_DATASET.data), 1);
    assert.strictEqual(dt3.evaluate(OBJECT_EVALUATION_DATASET.data), 1);
  });

  /**
   * https://github.com/serendipious/nodejs-decision-tree/issues/22
   */
  it('should be able to export and import a trained model', () => {
    const dt1 = new DecisionTree(TIC_TAC_TOE_DATASET.data, 'classification', TIC_TAC_TOE_DATASET.features);
    const dt1ExportedModelJSON = dt1.toJSON();
    
    dt1.import(dt1ExportedModelJSON);
    assert.strictEqual(dt1.evaluate(TIC_TAC_TOE_DATASET.data), 1);
  });
});
