import { strict as assert } from 'assert';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import DecisionTree from '../lib/decision-tree.js';

// Helper function to load JSON files
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadJSON(filename) {
  const filePath = join(__dirname, '..', 'data', filename);
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

const OBJECT_EVALUATION_DATASET = loadJSON('object-evaluation.json');
const TIC_TAC_TOE_DATASET = loadJSON('tic-tac-toe.json');
const VOTING_DATASET = loadJSON('voting.json');

describe('DecisionTree Decision Tree on Sample Datasets', function() {
  describe('Tic Tac Toe Dataset', function() {
    const dt = new DecisionTree(TIC_TAC_TOE_DATASET.data, 'classification', TIC_TAC_TOE_DATASET.features);

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
    const dt = new DecisionTree(VOTING_DATASET.data, 'classification', VOTING_DATASET.features);
    
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
    const dt = new DecisionTree(OBJECT_EVALUATION_DATASET.data, 'classification', OBJECT_EVALUATION_DATASET.features);

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
