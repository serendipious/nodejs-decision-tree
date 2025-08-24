"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var _ = __importStar(require("lodash"));
// Node types constant
var NODE_TYPES = {
    RESULT: 'result',
    FEATURE: 'feature',
    FEATURE_VALUE: 'feature_value'
};
/**
 * Decision Tree class implementing ID3 algorithm
 */
var DecisionTree = /** @class */ (function () {
    function DecisionTree() {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
        this.data = [];
        var numArgs = args.length;
        if (numArgs === 1) {
            this.import(args[0]);
        }
        else if (numArgs === 2) {
            var target = args[0], features = args[1];
            if (!target || typeof target !== 'string') {
                throw new Error('`target` argument is expected to be a String. Check documentation on usage');
            }
            if (!features || !Array.isArray(features)) {
                throw new Error('`features` argument is expected to be an Array<String>. Check documentation on usage');
            }
            this.target = target;
            this.features = features;
        }
        else if (numArgs === 3) {
            var data = args[0], target = args[1], features = args[2];
            var instance = new DecisionTree(target, features);
            instance.train(data);
            return instance;
        }
        else {
            throw new Error('Invalid arguments passed to constructor. Check documentation on usage');
        }
    }
    /**
     * Trains the decision tree with provided data
     * @param data - Array of training data objects
     */
    DecisionTree.prototype.train = function (data) {
        if (!data || !Array.isArray(data)) {
            throw new Error('`data` argument is expected to be an Array<Object>. Check documentation on usage');
        }
        this.model = createTree(data, this.target, this.features);
    };
    /**
     * Predicts class for a given sample
     * @param sample - Sample data to predict
     * @returns Predicted class value
     */
    DecisionTree.prototype.predict = function (sample) {
        var root = this.model;
        var _loop_1 = function () {
            var attr = root.name;
            var sampleVal = sample[attr];
            var childNode = _.find(root.vals, function (node) {
                return node.name == sampleVal;
            });
            if (childNode) {
                root = childNode.child;
            }
            else {
                root = root.vals[0].child;
            }
        };
        while (root.type !== NODE_TYPES.RESULT) {
            _loop_1();
        }
        return root.val;
    };
    /**
     * Evaluates prediction accuracy on samples
     * @param samples - Array of test samples
     * @returns Accuracy ratio (correct predictions / total predictions)
     */
    DecisionTree.prototype.evaluate = function (samples) {
        var _this = this;
        var total = 0;
        var correct = 0;
        _.each(samples, function (s) {
            total++;
            var pred = _this.predict(s);
            var actual = s[_this.target];
            if (_.isEqual(pred, actual)) {
                correct++;
            }
        });
        return correct / total;
    };
    /**
     * Imports a previously saved model with the toJSON() method
     * @param json - JSON representation of the model
     */
    DecisionTree.prototype.import = function (json) {
        var model = json.model, data = json.data, target = json.target, features = json.features;
        this.model = model;
        this.data = data;
        this.target = target;
        this.features = features;
    };
    /**
     * Returns JSON representation of trained model
     * @returns JSON object containing model data
     */
    DecisionTree.prototype.toJSON = function () {
        var _a = this, data = _a.data, target = _a.target, features = _a.features;
        var model = this.model;
        return { model: model, data: data, target: target, features: features };
    };
    DecisionTree.NODE_TYPES = NODE_TYPES;
    return DecisionTree;
}());
/**
 * Creates a new tree using ID3 algorithm
 * @private
 */
function createTree(data, target, features) {
    var targets = _.uniq(_.map(data, target));
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
    var bestFeatureName = bestFeature.name;
    var bestFeatureGain = bestFeature.gain;
    var remainingFeatures = _.without(features, bestFeatureName);
    var possibleValues = _.uniq(_.map(data, bestFeatureName));
    var node = {
        name: bestFeatureName,
        alias: bestFeatureName + randomUUID(),
        gain: bestFeatureGain,
        sampleSize: data.length,
        type: NODE_TYPES.FEATURE,
        vals: _.map(possibleValues, function (featureVal) {
            var featureValDataSample = data.filter(function (dataRow) { return dataRow[bestFeatureName] == featureVal; });
            var featureValDataSampleSize = featureValDataSample.length;
            var child_node = {
                name: featureVal,
                alias: featureVal + randomUUID(),
                type: NODE_TYPES.FEATURE_VALUE,
                prob: featureValDataSampleSize / data.length,
                sampleSize: featureValDataSampleSize
            };
            child_node.child = createTree(featureValDataSample, target, remainingFeatures);
            return child_node;
        })
    };
    return node;
}
/**
 * Computes entropy of a list
 * @private
 */
function entropy(vals) {
    var uniqueVals = _.uniq(vals);
    var probs = uniqueVals.map(function (x) {
        return prob(x, vals);
    });
    var logVals = probs.map(function (p) {
        return -p * log2(p);
    });
    return logVals.reduce(function (a, b) {
        return a + b;
    }, 0);
}
/**
 * Computes information gain
 * @private
 */
function gain(data, target, feature) {
    var attrVals = _.uniq(_.map(data, feature));
    var setEntropy = entropy(_.map(data, target));
    var setSize = _.size(data);
    var entropies = attrVals.map(function (n) {
        var subset = data.filter(function (x) {
            return x[feature] === n;
        });
        return (subset.length / setSize) * entropy(_.map(subset, target));
    });
    var sumOfEntropies = entropies.reduce(function (a, b) {
        return a + b;
    }, 0);
    return setEntropy - sumOfEntropies;
}
/**
 * Computes Max gain across features to determine best split
 * @private
 */
function maxGain(data, target, features) {
    var maxGain;
    var maxGainFeature;
    for (var _i = 0, features_1 = features; _i < features_1.length; _i++) {
        var feature = features_1[_i];
        var featureGain = gain(data, target, feature);
        if (!maxGain || maxGain < featureGain) {
            maxGain = featureGain;
            maxGainFeature = feature;
        }
    }
    return { gain: maxGain, name: maxGainFeature };
}
/**
 * Computes probability of a given value existing in a given list
 * @private
 */
function prob(value, list) {
    var occurrences = _.filter(list, function (element) {
        return element === value;
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
    list.forEach(function (element) {
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
module.exports = DecisionTree;
//# sourceMappingURL=decision-tree.js.map