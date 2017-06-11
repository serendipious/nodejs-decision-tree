var _ = require('underscore');
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
    // added probability, gain, sample size details for more insights
    // set root level probability multipler 
    model = createTree(data, target, features, 1, 1);
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
        var childNode = _.find(root.vals, function(node) {
          return node.name == sampleVal
        });
        if (childNode){
          root = childNode.child;
        } else {
          root = root.vals[0].child;
        }
      }

      return root;
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
        if (_.isEqual(pred, actual)) {
          correct++;
        }
      });

      return correct / total;
    },
    
    /**
     * Imports a previously saved model with the toJSON() method
     */
    import: function(json) {
      model = json;
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
  function createTree(data, target, features,probabilityMultipler,gainMultipler) {
    //console.log(data.length);
    var targets = _.uniq(_.pluck(data, target));
    //console.log(targets);
    var sample = _.countBy(data, target);
    if (targets.length == 1) {
      return {
        type: NODE_TYPES.RESULT,
        val: targets[0],
        name: targets[0],
        alias: targets[0] + randomUUID(),
        probabilityMul: parseFloat(probabilityMultipler * 100).toFixed(2),
        probability: parseFloat(1 * 100).toFixed(2),
        total: data.length,
        sample: sample
      };
    }

    if (features.length == 0) {
      var topTarget = mostCommon(sample);
      return {
        type: NODE_TYPES.RESULT,
        val: topTarget.value,
        name: topTarget.value,
        alias: topTarget.value + randomUUID(),
        probabilityMul: parseFloat(topTarget.probability * probabilityMultipler * 100).toFixed(2),
        probability:  parseFloat(topTarget.probability * 100).toFixed(2),
        total: data.length,
        sample: sample
      };
    }

    var bestFeature = maxGain(data, target, features);
    var remainingFeatures = _.without(features, bestFeature.element);
    var possibleValues = _.uniq(_.pluck(data, bestFeature.element));
    var featuresSample = _.countBy(data, bestFeature.element);
    //console.log('****Best Feature****');
    //console.log(bestFeature);
    var node = {
      name: bestFeature.element,
      alias: bestFeature.element + randomUUID()
    };
    
    node.type = NODE_TYPES.FEATURE;
    node.gain = parseFloat(bestFeature.gain * 100).toFixed(2);
    node.gainMul = parseFloat(bestFeature.gain * gainMultipler * 100).toFixed(2);
    node.total = data.length;
    node.sample = featuresSample;

    node.vals = _.map(possibleValues, function(v) {
      var _newS = data.filter(function(x) {
        return x[bestFeature.element] == v
      });      
      var child_node = {
        name: v,
        alias: v + randomUUID(),
        type: NODE_TYPES.FEATURE_VALUE
      };
      var elementProbValue = featuresSample[v]/data.length;
      child_node.probabilityMul = parseFloat(elementProbValue * probabilityMultipler * 100).toFixed(2); 
      child_node.probability = parseFloat(elementProbValue * 100).toFixed(2);

      child_node.child = createTree(_newS, target, remainingFeatures,elementProbValue,bestFeature.gain);
      return child_node;
    });
    return node;
  }

  /**
   * Computes entropy of a list
   * @private
   */
  function entropy(vals) {
    var uniqVals = _.uniq(vals);
    var probs = uniqVals.map(function(x) {
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
    var attrVals = _.uniq(_.pluck(data, feature));
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
    // console.log('*****GAIN*****');
    // console.log('*****SetEntropy*****');    
    // console.log(setEntropy);
    // console.log('*****SumofEntropies*****');
    // console.log(sumOfEntropies);
    return setEntropy - sumOfEntropies;
  }

  /**
   * Computes Max gain across features to determine best split
   * @private
   */
  function maxGain(data, target, features) {
    var featureGain = []; 
    _.map(features, function(element) {
        // console.log('***gain**'); 
        // console.log(element);
        // console.log( gain(data, target, element));
        featureGain.push({ element:element, gain: gain(data, target, element)});
    });
    return _.max(featureGain,function(element){ return element.gain; });
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
    // console.log('*******ProbabilityValue***********');
    // console.log(value);
    // console.log(numOccurrences);
    // console.log(numElements);
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
    var largestFrequency = -1;
    var mostCommonElement = null;
    var result = {};
    
    for(var key in list)
    {
        var elementFrequency = list[key];
        if (largestFrequency < elementFrequency) {
          mostCommonElement = key;
          largestFrequency = elementFrequency;
        }
    }
      
    var numOccurrences = largestFrequency, 
    numElements = list.length;
    var probability = (numOccurrences/numElements)*100;
    
    result.value = mostCommonElement;
    result.probability = probability;
    // result.total = numElements;
    // result.sample = list;

    return result;
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
