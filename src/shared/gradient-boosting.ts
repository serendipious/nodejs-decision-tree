/**
 * Gradient boosting core algorithm for XGBoost
 */

import _ from 'lodash';
import { 
  TreeNode, 
  TrainingData, 
  XGBoostConfig, 
  GradientHessian, 
  WeightedSample,
  NODE_TYPES 
} from './types.js';
import { LossFunctionFactory } from './loss-functions.js';
import { SeededRandom, selectRandomFeatures } from './utils.js';
import { createTree } from './id3-algorithm.js';

/**
 * Weighted decision tree for gradient boosting
 */
export function createWeightedTree(
  data: TrainingData[], 
  target: string, 
  features: string[],
  weights: number[],
  gradients: number[],
  hessians: number[],
  config: XGBoostConfig
): TreeNode {
  const weightedData = data.map((sample, index) => ({
    ...sample,
    _weight: weights[index],
    _gradient: gradients[index],
    _hessian: hessians[index]
  }));

  return createWeightedTreeRecursive(
    weightedData, 
    target, 
    features, 
    config,
    0
  );
}

/**
 * Recursive weighted tree creation
 */
function createWeightedTreeRecursive(
  data: TrainingData[], 
  target: string, 
  features: string[],
  config: XGBoostConfig,
  currentDepth: number
): TreeNode {
  const maxDepth = config.maxDepth || 6;
  const minChildWeight = config.minChildWeight || 1;
  
  // Check stopping criteria
  if (currentDepth >= maxDepth || data.length === 0) {
    return createLeafNode(data, config);
  }

  // Check minimum child weight
  const totalWeight = data.reduce((sum, sample) => sum + (sample as any)._weight, 0);
  if (totalWeight < minChildWeight) {
    return createLeafNode(data, config);
  }

  // Find best split
  const bestSplit = findBestWeightedSplit(data, target, features, config);
  if (!bestSplit) {
    return createLeafNode(data, config);
  }

  // Create internal node
  const node: TreeNode = {
    name: bestSplit.feature,
    alias: bestSplit.feature + '_' + Math.random().toString(32).slice(2),
    gain: bestSplit.gain,
    sampleSize: data.length,
    type: NODE_TYPES.FEATURE,
    vals: []
  };

  // Split data and create child nodes
  const remainingFeatures = features.filter(f => f !== bestSplit.feature);
  const possibleValues = _.uniq(data.map(sample => sample[bestSplit.feature]));

  node.vals = possibleValues.map(value => {
    const childData = data.filter(sample => sample[bestSplit.feature] === value);
    const childNode: TreeNode = {
      name: value,
      alias: value + '_' + Math.random().toString(32).slice(2),
      type: NODE_TYPES.FEATURE_VALUE,
      prob: childData.length / data.length,
      sampleSize: childData.length
    };

    childNode.child = createWeightedTreeRecursive(
      childData, 
      target, 
      remainingFeatures, 
      config,
      currentDepth + 1
    );

    return childNode;
  });

  return node;
}

/**
 * Find best weighted split
 */
function findBestWeightedSplit(
  data: TrainingData[], 
  target: string, 
  features: string[],
  config: XGBoostConfig
): { feature: string; gain: number } | null {
  let bestGain = -Infinity;
  let bestFeature: string | null = null;

  for (const feature of features) {
    const gain = calculateWeightedGain(data, target, feature);
    if (gain > bestGain) {
      bestGain = gain;
      bestFeature = feature;
    }
  }

  return bestFeature ? { feature: bestFeature, gain: bestGain } : null;
}

/**
 * Calculate weighted information gain
 */
function calculateWeightedGain(
  data: TrainingData[], 
  target: string, 
  feature: string
): number {
  const uniqueValues = _.uniq(data.map(sample => sample[feature]));
  const totalWeight = data.reduce((sum, sample) => sum + (sample as any)._weight, 0);
  
  if (totalWeight === 0) return 0;

  // Calculate weighted entropy for each value
  let weightedEntropy = 0;
  for (const value of uniqueValues) {
    const subset = data.filter(sample => sample[feature] === value);
    const subsetWeight = subset.reduce((sum, sample) => sum + (sample as any)._weight, 0);
    
    if (subsetWeight > 0) {
      const entropy = calculateWeightedEntropy(subset, target);
      weightedEntropy += (subsetWeight / totalWeight) * entropy;
    }
  }

  const totalEntropy = calculateWeightedEntropy(data, target);
  return totalEntropy - weightedEntropy;
}

/**
 * Calculate weighted entropy
 */
function calculateWeightedEntropy(data: TrainingData[], target: string): number {
  const targetValues = data.map(sample => sample[target]);
  const uniqueValues = _.uniq(targetValues);
  
  let entropy = 0;
  for (const value of uniqueValues) {
    const subset = data.filter(sample => sample[target] === value);
    const subsetWeight = subset.reduce((sum, sample) => sum + (sample as any)._weight, 0);
    const totalWeight = data.reduce((sum, sample) => sum + (sample as any)._weight, 0);
    
    if (totalWeight > 0) {
      const probability = subsetWeight / totalWeight;
      if (probability > 0) {
        entropy -= probability * Math.log2(probability);
      }
    }
  }
  
  return entropy;
}

/**
 * Create leaf node with weighted prediction
 */
function createLeafNode(data: TrainingData[], config: XGBoostConfig): TreeNode {
  const regLambda = config.regLambda || 1;
  
  // Calculate weighted average of gradients/hessians
  let gradientSum = 0;
  let hessianSum = 0;
  let totalWeight = 0;

  for (const sample of data) {
    const weight = (sample as any)._weight || 1;
    const gradient = (sample as any)._gradient || 0;
    const hessian = (sample as any)._hessian || 1;
    
    gradientSum += weight * gradient;
    hessianSum += weight * hessian;
    totalWeight += weight;
  }

  // Leaf value = -gradient_sum / (hessian_sum + lambda)
  const leafValue = hessianSum > 0 ? -gradientSum / (hessianSum + regLambda) : 0;

  return {
    type: NODE_TYPES.RESULT,
    val: leafValue,
    name: leafValue.toString(),
    alias: 'leaf_' + Math.random().toString(32).slice(2)
  };
}

/**
 * Create weighted sample for training
 */
export function createWeightedSample(
  data: TrainingData[],
  config: XGBoostConfig,
  random: SeededRandom
): WeightedSample {
  const subsample = config.subsample || 1;
  const colsampleByTree = config.colsampleByTree || 1;
  
  // Subsample data
  let sampledData = data;
  if (subsample < 1) {
    const sampleSize = Math.floor(data.length * subsample);
    const indices = Array.from({ length: data.length }, (_, i) => i);
    const sampledIndices: number[] = [];
    
    for (let i = 0; i < sampleSize; i++) {
      const randomIndex = random.nextInt(indices.length);
      sampledIndices.push(indices[randomIndex]);
      indices.splice(randomIndex, 1);
    }
    
    sampledData = sampledIndices.map(i => data[i]);
  }

  // Sample features
  let allFeatures: string[] = [];
  if (sampledData.length > 0) {
    allFeatures = Object.keys(sampledData[0]).filter(key => key !== 'target');
  }
  const selectedFeatures = selectRandomFeatures(
    allFeatures, 
    Math.max(1, Math.floor(allFeatures.length * colsampleByTree)), 
    random
  );

  return {
    data: sampledData,
    weights: new Array(sampledData.length).fill(1),
    gradients: new Array(sampledData.length).fill(0),
    hessians: new Array(sampledData.length).fill(1)
  };
}

/**
 * Calculate base score for initial prediction
 */
export function calculateBaseScore(
  data: TrainingData[], 
  target: string, 
  objective: 'regression' | 'binary' | 'multiclass'
): number {
  const targetValues = data.map(sample => sample[target]);
  
  switch (objective) {
    case 'regression':
      return targetValues.reduce((sum, val) => sum + val, 0) / targetValues.length;
    
    case 'binary':
      const positiveCount = targetValues.filter(val => val === 1 || val === true).length;
      const probability = positiveCount / targetValues.length;
      return Math.log(probability / (1 - probability + 1e-15));
    
    case 'multiclass':
      return 0; // Will be handled differently for multiclass
    
    default:
      return 0;
  }
}
