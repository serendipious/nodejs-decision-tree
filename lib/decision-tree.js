var _ = require('lodash');

/**
 * ID3 Decision Tree Algorithm
 * @module DecisionTreeID3
 */

module.exports = (function() {

  /**
   * Map of valid tree node types
   * @constant
   * @static
   */
  const NODE_TYPES = DecisionTreeID3.NODE_TYPES = {
    RESULT: 'result',
    FEATURE: 'feature',
    FEATURE_VALUE: 'feature_value'
  };

  /**
   * Underlying model
   * @private
   */
  var model;

  /**
   * @constructor
   * @return {DecisionTreeID3}
   */
  function DecisionTreeID3(data, target, features) {
    this.data = data;
    this.target = target;
    this.features = features;
    model = createTree(data, target, features);
  }

  /**
   * @public API
   */
  DecisionTreeID3.prototype = {

    /**
     * Predicts class for sample
     */
    predict: function(sample) {
      var root = model;
      while (root.type !== NODE_TYPES.RESULT) {
        var attr = root.name;
        var sampleVal = sample[attr];
        var childNode = _.detect(root.vals, function(node) {
          return node.name == sampleVal
        });
        if (childNode){
          root = childNode.child;
        } else {
          root = root.vals[0].child;
        }
      }

      return root.val;
    },

    /**
     * Evalutes prediction accuracy on samples
     */
    evaluate: function(samples) {
      var instance = this;
      var target = this.target;

      var total = 0;
      var correct = 0;

      _.each(samples, function(s) {
        total++;
        var pred = instance.predict(s);
        var actual = s[target];
        if (pred == actual) {
          correct++;
        }
      });

      return correct / total;
    },

    /**
     * Returns JSON representation of trained model
     */
    toJSON: function() {
      return model;
    }
  };

  /**
   * Creates a new tree
   * @private
   */
  function createTree(data, target, features) {
    var targets = _.unique(_.pluck(data, target));
    if (targets.length == 1) {
      return {
        type: NODE_TYPES.RESULT,
        val: targets[0],
        name: targets[0],
        alias: targets[0] + randomUUID()
      };
    }

    if (features.length == 0) {
      var topTarget = mostCommon(targets);
      return {
        type: NODE_TYPES.RESULT,
        val: topTarget,
        name: topTarget,
        alias: topTarget + randomUUID()
      };
    }

    var bestFeature = maxGain(data, target, features);
    var remainingFeatures = _.without(features, bestFeature);
    var possibleValues = _.unique(_.pluck(data, bestFeature));
    
    var node = {
      name: bestFeature,
      alias: bestFeature + randomUUID()
    };
    
    node.type = NODE_TYPES.FEATURE;
    node.vals = _.map(possibleValues, function(v) {
      var _newS = data.filter(function(x) {
        return x[bestFeature] == v
      });

      var child_node = {
        name: v,
        alias: v + randomUUID(),
        type: NODE_TYPES.FEATURE_VALUE
      };

      child_node.child = createTree(_newS, target, remainingFeatures);
      return child_node;
    });

    return node;
  }

  /**
   * Computes entropy of a list
   * @private
   */
  function entropy(vals) {
    var uniqueVals = _.unique(vals);
    var probs = uniqueVals.map(function(x) {
      return prob(x, vals)
    });

    var logVals = probs.map(function(p) {
      return -p * log2(p)
    });
    
    return logVals.reduce(function(a, b) {
      return a + b
    }, 0);
  }

  /**
   * Computes gain
   * @private
   */
  function gain(data, target, feature) {
    var attrVals = _.unique(_.pluck(data, feature));
    var setEntropy = entropy(_.pluck(data, target));
    var setSize = _.size(data);
    
    var entropies = attrVals.map(function(n) {
      var subset = data.filter(function(x) {
        return x[feature] === n
      });

      return (subset.length / setSize) * entropy(_.pluck(subset, target));
    });

    var sumOfEntropies = entropies.reduce(function(a, b) {
      return a + b
    }, 0);
    
    return setEntropy - sumOfEntropies;
  }

  /**
   * Computes Max gain across features to determine best split
   * @private
   */
  function maxGain(data, target, features) {
    return _.max(features, function(element) {
      return gain(data, target, element)
    });
  }

  /**
   * Computes probability of of a given value existing in a given list
   * @private
   */
  function prob(value, list) {
    var occurrences = _.filter(list, function(element) {
      return element === value
    });

    var numOccurrences = occurrences.length;
    var numElements = list.length;
    return numOccurrences / numElements;
  }

  /**
   * Computes Log with base-2
   * @private
   */
  function log2(n) {
    return Math.log(n) / Math.log(2);
  }

  /**
   * Finds element with highest occurrence in a list
   * @private
   */
  function mostCommon(list) {
    var elementFrequencyMap = {};
    var largestFrequency = -1;
    var mostCommonElement = null;

    list.forEach(function(element) {
      var elementFrequency = (elementFrequencyMap[element] || 0) + 1;
      elementFrequencyMap[element] = elementFrequency;

      if (largestFrequency < elementFrequency) {
        mostCommonElement = element;
        largestFrequency = elementFrequency;
      }
    });

    return mostCommonElement;
  }

  /**
   * Generates random UUID
   * @private
   */
  function randomUUID() {
    return "_r" + Math.random().toString(32).slice(2);
  }

  /**
   * @class DecisionTreeID3
   */
  return DecisionTreeID3;
})();
