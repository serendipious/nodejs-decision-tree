const assert = require('assert');
const _ = require('lodash');

/**
 * ID3 Decision Tree Algorithm
 * @module DecisionTreeID3
 */

module.exports = (function () {

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
   * @constructor
   * @return {DecisionTreeID3}
   */
  function DecisionTreeID3() {
    const numArgs = arguments.length;
    if (numArgs === 1) {
      this.import(arguments[0]);
    }
    else if (numArgs === 2) {
      const [target, features] = arguments;
      
      assert.ok(!!target && target.constructor === String, '`target` argument is expected to be a String. Check documentation on usage');
      assert.ok(!!features && features.constructor === Array, '`features` argument is expected to be an Array<String>. Check documentation on usage');

      this.target = target;
      this.features = features;
    }
    else if (numArgs === 3) {
      const [data, target, features] = arguments;
      const instance = new DecisionTreeID3(target, features);
      instance.train(data);
      return instance;
    }
    else {
      throw new Error('Invalid arguments passed to constructor. Check documentation on usage');
    }
  }

  /**
   * Decision Tree API
   */
  DecisionTreeID3.prototype = {
    /**
     * 
     * @param {Array<Object>} data
     */
    train: function(data) {
      assert.ok(!!data && data.constructor === Array, '`data` argument is expected to be an Array<Object>. Check documentation on usage');
      this.model = createTree(data, this.target, this.features);
    },

    /**
     * Predicts class for sample
     */
    predict: function (sample) {
      let root = this.model;
      while (root.type !== NODE_TYPES.RESULT) {
        let attr = root.name;
        let sampleVal = sample[attr];
        let childNode = _.find(root.vals, function (node) {
          return node.name == sampleVal
        });
        if (childNode) {
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
    evaluate: function (samples) {
      let instance = this;
      let target = this.target;

      let total = 0;
      let correct = 0;

      _.each(samples, function (s) {
        total++;
        let pred = instance.predict(s);
        let actual = s[target];
        if (_.isEqual(pred, actual)) {
          correct++;
        }
      });

      return correct / total;
    },

    /**
     * Imports a previously saved model with the toJSON() method
     */
    import: function (json) {
      var {model, data, target, features} = json;

      this.model = model;
      this.data = data;
      this.target = target;
      this.features = features;
    },

    /**
     * Returns JSON representation of trained model
     */
    toJSON: function () {
      var {data, target, features} = this;
      var model = this.model;

      return {model, data, target, features};
    }
  };

  /**
   * Creates a new tree
   * @private
   */
  function createTree(data, target, features) {
    let targets = _.uniq(_.map(data, target));
    if (targets.length == 1) {
      return {
        type: NODE_TYPES.RESULT,
        val: targets[0],
        name: targets[0],
        alias: targets[0] + randomUUID()
      };
    }

    if (features.length == 0) {
      let topTarget = mostCommon(targets);
      return {
        type: NODE_TYPES.RESULT,
        val: topTarget,
        name: topTarget,
        alias: topTarget + randomUUID()
      };
    }

    let bestFeature = maxGain(data, target, features);
    let bestFeatureName = bestFeature.name;
    let bestFeatureGain = bestFeature.gain;
    let remainingFeatures = _.without(features, bestFeatureName);
    let possibleValues = _.uniq(_.map(data, bestFeatureName));

    let node = {
      name: bestFeatureName,
      alias: bestFeatureName + randomUUID(),
      gain: bestFeatureGain,
      sampleSize: data.length
    };

    node.type = NODE_TYPES.FEATURE;
    node.vals = _.map(possibleValues, function (featureVal) {
      const featureValDataSample = data.filter((dataRow) => dataRow[bestFeatureName] == featureVal);
      const featureValDataSampleSize = featureValDataSample.length;

      const child_node = {
        name: featureVal,
        alias: featureVal + randomUUID(),
        type: NODE_TYPES.FEATURE_VALUE,
        prob: featureValDataSampleSize / data.length,
        sampleSize: featureValDataSampleSize
      };

      child_node.child = createTree(featureValDataSample, target, remainingFeatures);
      return child_node;
    });

    return node;
  }

  /**
   * Computes entropy of a list
   * @private
   */
  function entropy(vals) {
    let uniqueVals = _.uniq(vals);
    let probs = uniqueVals.map(function (x) {
      return prob(x, vals)
    });

    let logVals = probs.map(function (p) {
      return -p * log2(p)
    });

    return logVals.reduce(function (a, b) {
      return a + b
    }, 0);
  }

  /**
   * Computes gain
   * @private
   */
  function gain(data, target, feature) {
    let attrVals = _.uniq(_.map(data, feature));
    let setEntropy = entropy(_.map(data, target));
    let setSize = _.size(data);

    let entropies = attrVals.map(function (n) {
      let subset = data.filter(function (x) {
        return x[feature] === n
      });

      return (subset.length / setSize) * entropy(_.map(subset, target));
    });

    let sumOfEntropies = entropies.reduce(function (a, b) {
      return a + b
    }, 0);

    return setEntropy - sumOfEntropies;
  }

  /**
   * Computes Max gain across features to determine best split
   * @private
   */
  function maxGain(data, target, features) {
    let maxGain, maxGainFeature;
    for (let feature of features) {
      const featureGain = gain(data, target, feature);
      if (!maxGain || maxGain < featureGain) {
        maxGain = featureGain;
        maxGainFeature = feature;
      }
    }
    return {gain: maxGain, name: maxGainFeature};
  }

  /**
   * Computes probability of of a given value existing in a given list
   * @private
   */
  function prob(value, list) {
    let occurrences = _.filter(list, function (element) {
      return element === value
    });

    let numOccurrences = occurrences.length;
    let numElements = list.length;
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
    let elementFrequencyMap = {};
    let largestFrequency = -1;
    let mostCommonElement = null;

    list.forEach(function (element) {
      let elementFrequency = (elementFrequencyMap[element] || 0) + 1;
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
