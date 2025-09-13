/**
 * ID3 Algorithm implementation shared between Decision Tree and Random Forest
 */

import _ from 'lodash';
import { TreeNode, TrainingData, FeatureGain, NODE_TYPES } from './types.js';
import { randomUUID, prob, log2, mostCommon } from './utils.js';

/**
 * Creates a new tree using ID3 algorithm
 * @private
 */
export function createTree(
  data: TrainingData[], 
  target: string, 
  features: string[],
  maxDepth?: number,
  minSamplesSplit?: number,
  currentDepth: number = 0
): TreeNode {
  let targets = _.uniq(_.map(data, target));
  
  // Base case: all samples have same target
  if (targets.length == 1) {
    return {
      type: NODE_TYPES.RESULT,
      val: targets[0],
      name: targets[0],
      alias: targets[0] + randomUUID()
    };
  }

  // Base case: no features left
  if (features.length == 0) {
    let topTarget = mostCommon(targets);
    return {
      type: NODE_TYPES.RESULT,
      val: topTarget,
      name: topTarget,
      alias: topTarget + randomUUID()
    };
  }

  // Base case: max depth reached
  if (maxDepth && currentDepth >= maxDepth) {
    let topTarget = mostCommon(targets);
    return {
      type: NODE_TYPES.RESULT,
      val: topTarget,
      name: topTarget,
      alias: topTarget + randomUUID()
    };
  }

  // Base case: not enough samples to split
  if (minSamplesSplit && data.length < minSamplesSplit) {
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

      child_node.child = createTree(
        featureValDataSample, 
        target, 
        remainingFeatures,
        maxDepth,
        minSamplesSplit,
        currentDepth + 1
      );
      return child_node;
    })
  };

  return node;
}

/**
 * Computes entropy of a list
 * @private
 */
export function entropy(vals: any[]): number {
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
export function gain(data: TrainingData[], target: string, feature: string): number {
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
export function maxGain(data: TrainingData[], target: string, features: string[]): FeatureGain {
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
