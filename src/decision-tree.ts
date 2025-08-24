import * as _ from 'lodash';

/**
 * Decision Tree Algorithm
 * @module DecisionTree
 */

// Type definitions
interface TreeNode {
  type: string;
  name: string;
  alias: string;
  val?: any;
  gain?: number;
  sampleSize?: number;
  vals?: TreeNode[];
  child?: TreeNode;
  prob?: number;
}

interface DecisionTreeData {
  model: TreeNode;
  data: any[];
  target: string;
  features: string[];
}

interface FeatureGain {
  gain: number;
  name: string;
}

interface TrainingData {
  [key: string]: any;
}

// Node types constant
const NODE_TYPES = {
  RESULT: 'result',
  FEATURE: 'feature',
  FEATURE_VALUE: 'feature_value'
} as const;

type NodeType = typeof NODE_TYPES[keyof typeof NODE_TYPES];

/**
 * Decision Tree class implementing ID3 algorithm
 */
class DecisionTree {
  public static readonly NODE_TYPES = NODE_TYPES;
  
  private model!: TreeNode;
  private data: any[] = [];
  private target!: string;
  private features!: string[];

  constructor(...args: any[]) {
    const numArgs = args.length;
    
    if (numArgs === 1) {
      this.import(args[0]);
    }
    else if (numArgs === 2) {
      const [target, features] = args;
      
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
      const [data, target, features] = args;
      const instance = new DecisionTree(target, features);
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
  train(data: TrainingData[]): void {
    if (!data || !Array.isArray(data)) {
      throw new Error('`data` argument is expected to be an Array<Object>. Check documentation on usage');
    }
    this.model = createTree(data, this.target, this.features);
  }

  /**
   * Predicts class for a given sample
   * @param sample - Sample data to predict
   * @returns Predicted class value
   */
  predict(sample: TrainingData): any {
    let root = this.model;
    while (root.type !== NODE_TYPES.RESULT) {
      let attr = root.name;
      let sampleVal = sample[attr];
      let childNode = _.find(root.vals, function (node) {
        return node.name == sampleVal;
      });
      if (childNode) {
        root = childNode.child!;
      } else {
        root = root.vals![0].child!;
      }
    }

    return root.val;
  }

  /**
   * Evaluates prediction accuracy on samples
   * @param samples - Array of test samples
   * @returns Accuracy ratio (correct predictions / total predictions)
   */
  evaluate(samples: TrainingData[]): number {
    let total = 0;
    let correct = 0;

    _.each(samples, (s) => {
      total++;
      let pred = this.predict(s);
      let actual = s[this.target];
      if (_.isEqual(pred, actual)) {
        correct++;
      }
    });

    return correct / total;
  }

  /**
   * Imports a previously saved model with the toJSON() method
   * @param json - JSON representation of the model
   */
  import(json: DecisionTreeData): void {
    const {model, data, target, features} = json;

    this.model = model;
    this.data = data;
    this.target = target;
    this.features = features;
  }

  /**
   * Returns JSON representation of trained model
   * @returns JSON object containing model data
   */
  toJSON(): DecisionTreeData {
    const {data, target, features} = this;
    const model = this.model;

    return {model, data, target, features};
  }
}

/**
 * Creates a new tree using ID3 algorithm
 * @private
 */
function createTree(data: TrainingData[], target: string, features: string[]): TreeNode {
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

  let node: TreeNode = {
    name: bestFeatureName,
    alias: bestFeatureName + randomUUID(),
    gain: bestFeatureGain,
    sampleSize: data.length,
    type: NODE_TYPES.FEATURE,
    vals: _.map(possibleValues, function (featureVal) {
      const featureValDataSample = data.filter((dataRow) => dataRow[bestFeatureName] == featureVal);
      const featureValDataSampleSize = featureValDataSample.length;

      const child_node: TreeNode = {
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
function entropy(vals: any[]): number {
  let uniqueVals = _.uniq(vals);
  let probs = uniqueVals.map(function (x) {
    return prob(x, vals);
  });

  let logVals = probs.map(function (p) {
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
function gain(data: TrainingData[], target: string, feature: string): number {
  let attrVals = _.uniq(_.map(data, feature));
  let setEntropy = entropy(_.map(data, target));
  let setSize = _.size(data);

  let entropies = attrVals.map(function (n) {
    let subset = data.filter(function (x) {
      return x[feature] === n;
    });

    return (subset.length / setSize) * entropy(_.map(subset, target));
  });

  let sumOfEntropies = entropies.reduce(function (a, b) {
    return a + b;
  }, 0);

  return setEntropy - sumOfEntropies;
}

/**
 * Computes Max gain across features to determine best split
 * @private
 */
function maxGain(data: TrainingData[], target: string, features: string[]): FeatureGain {
  let maxGain: number | undefined;
  let maxGainFeature: string | undefined;
  
  for (let feature of features) {
    const featureGain = gain(data, target, feature);
    if (!maxGain || maxGain < featureGain) {
      maxGain = featureGain;
      maxGainFeature = feature;
    }
  }
  
  return {gain: maxGain!, name: maxGainFeature!};
}

/**
 * Computes probability of a given value existing in a given list
 * @private
 */
function prob(value: any, list: any[]): number {
  let occurrences = _.filter(list, function (element) {
    return element === value;
  });

  let numOccurrences = occurrences.length;
  let numElements = list.length;
  return numOccurrences / numElements;
}

/**
 * Computes Log with base-2
 * @private
 */
function log2(n: number): number {
  return Math.log(n) / Math.log(2);
}

/**
 * Finds element with highest occurrence in a list
 * @private
 */
function mostCommon(list: any[]): any {
  let elementFrequencyMap: { [key: string]: number } = {};
  let largestFrequency = -1;
  let mostCommonElement: any = null;

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
function randomUUID(): string {
  return "_r" + Math.random().toString(32).slice(2);
}

// Export the DecisionTree class
export = DecisionTree;
