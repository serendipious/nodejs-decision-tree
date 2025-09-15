/**
 * CART (Classification and Regression Trees) Algorithm
 * Handles both discrete and continuous variables with optimal splitting
 */

import _ from 'lodash';
import { TreeNode, TrainingData, FeatureGain, NODE_TYPES } from './types.js';
import { randomUUID, log2 } from './utils.js';

export interface CARTSplitInfo {
  feature: string;
  threshold?: number;
  operator?: 'lte' | 'gt' | 'eq';
  gain: number;
  leftData: TrainingData[];
  rightData: TrainingData[];
  leftCount: number;
  rightCount: number;
}

export interface CARTConfig {
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxDepth?: number;
  maxFeatures?: number | 'sqrt' | 'log2' | 'auto';
  randomState?: number;
  criterion: 'gini' | 'entropy' | 'mse' | 'mae';
  continuousSplitting: 'binary' | 'multiway';
}

export class CARTAlgorithm {
  private config: CARTConfig;

  constructor(config: Partial<CARTConfig> = {}) {
    this.config = {
      minSamplesSplit: 2,
      minSamplesLeaf: 1,
      maxDepth: undefined,
      maxFeatures: 'auto',
      randomState: undefined,
      criterion: 'gini',
      continuousSplitting: 'binary',
      ...config
    };
  }

  /**
   * Creates a tree using CART algorithm
   * @param data - Training data
   * @param target - Target variable name
   * @param features - Available features
   * @param featureTypes - Data type information for features
   * @param currentDepth - Current tree depth
   * @returns Root tree node
   */
  createTree(
    data: TrainingData[],
    target: string,
    features: string[],
    featureTypes: { [feature: string]: 'discrete' | 'continuous' },
    currentDepth: number = 0
  ): TreeNode {
    // Base case: empty data
    if (data.length === 0) {
      throw new Error('Cannot create tree from empty dataset');
    }
    
    const targetValues = data.map(row => row[target]);
    
    // Base case: all samples have same target
    if (this.allSameValue(targetValues)) {
      return this.createLeafNode(this.getLeafValue(targetValues), data.length);
    }

    // Base case: no features left
    if (features.length === 0) {
      return this.createLeafNode(this.getLeafValue(targetValues), data.length);
    }

    // Base case: max depth reached
    if (this.config.maxDepth && currentDepth >= this.config.maxDepth) {
      return this.createLeafNode(this.getLeafValue(targetValues), data.length);
    }

    // Base case: not enough samples to split
    if (data.length < this.config.minSamplesSplit) {
      return this.createLeafNode(this.getLeafValue(targetValues), data.length);
    }

    // Find best split
    const bestSplit = this.findBestSplit(data, target, features, featureTypes);
    
    if (!bestSplit || bestSplit.gain <= 0) {
      return this.createLeafNode(this.getLeafValue(targetValues), data.length);
    }

    // Create internal node
    const node: TreeNode = {
      type: NODE_TYPES.FEATURE,
      name: bestSplit.feature,
      alias: bestSplit.feature + randomUUID(),
      gain: bestSplit.gain,
      sampleSize: data.length,
      vals: [],
      // Add continuous variable properties
      splitThreshold: bestSplit.threshold,
      splitOperator: bestSplit.operator
    };

    // Create child nodes based on split type
    if (featureTypes[bestSplit.feature] === 'continuous') {
      // Continuous feature - binary split
      const leftChild = this.createTree(
        bestSplit.leftData,
        target,
        features.filter(f => f !== bestSplit.feature),
        featureTypes,
        currentDepth + 1
      );

      const rightChild = this.createTree(
        bestSplit.rightData,
        target,
        features.filter(f => f !== bestSplit.feature),
        featureTypes,
        currentDepth + 1
      );

      // Create split value nodes
      node.vals = [
        {
          type: NODE_TYPES.FEATURE_VALUE,
          name: `<=${bestSplit.threshold}`,
          alias: `lte_${bestSplit.threshold}_${randomUUID()}`,
          prob: bestSplit.leftCount / data.length,
          sampleSize: bestSplit.leftCount,
          child: leftChild
        },
        {
          type: NODE_TYPES.FEATURE_VALUE,
          name: `>${bestSplit.threshold}`,
          alias: `gt_${bestSplit.threshold}_${randomUUID()}`,
          prob: bestSplit.rightCount / data.length,
          sampleSize: bestSplit.rightCount,
          child: rightChild
        }
      ];
    } else {
      // Discrete feature - multiway split
      const uniqueValues = [...new Set(data.map(row => row[bestSplit.feature]))];
      
      node.vals = uniqueValues.map(value => {
        const subsetData = data.filter(row => row[bestSplit.feature] === value);
        const child = this.createTree(
          subsetData,
          target,
          features.filter(f => f !== bestSplit.feature),
          featureTypes,
          currentDepth + 1
        );

        return {
          type: NODE_TYPES.FEATURE_VALUE,
          name: String(value),
          alias: String(value) + randomUUID(),
          prob: subsetData.length / data.length,
          sampleSize: subsetData.length,
          child
        };
      });
    }

    return node;
  }

  /**
   * Finds the best split for the given data
   */
  private findBestSplit(
    data: TrainingData[],
    target: string,
    features: string[],
    featureTypes: { [feature: string]: 'discrete' | 'continuous' }
  ): CARTSplitInfo | null {
    let bestSplit: CARTSplitInfo | null = null;
    let bestGain = -Infinity;

    for (const feature of features) {
      const split = this.findBestSplitForFeature(data, target, feature, featureTypes[feature]);
      
      if (split && split.gain > bestGain) {
        bestGain = split.gain;
        bestSplit = split;
      }
    }

    return bestSplit;
  }

  /**
   * Finds the best split for a specific feature
   */
  private findBestSplitForFeature(
    data: TrainingData[],
    target: string,
    feature: string,
    featureType: 'discrete' | 'continuous'
  ): CARTSplitInfo | null {
    if (featureType === 'continuous') {
      return this.findBestContinuousSplit(data, target, feature);
    } else {
      return this.findBestDiscreteSplit(data, target, feature);
    }
  }

  /**
   * Finds the best split for a continuous feature
   */
  private findBestContinuousSplit(
    data: TrainingData[],
    target: string,
    feature: string
  ): CARTSplitInfo | null {
    const values = data.map(row => Number(row[feature])).filter(n => !isNaN(n));
    
    if (values.length === 0) return null;

    const sortedValues = [...new Set(values)].sort((a, b) => a - b);
    const targetValues = data.map(row => row[target]);
    
    let bestSplit: CARTSplitInfo | null = null;
    let bestGain = -Infinity;

    // Test each possible threshold
    for (let i = 0; i < sortedValues.length - 1; i++) {
      const threshold = (sortedValues[i] + sortedValues[i + 1]) / 2;
      
      const leftData = data.filter(row => Number(row[feature]) <= threshold);
      const rightData = data.filter(row => Number(row[feature]) > threshold);
      
      if (leftData.length < this.config.minSamplesLeaf || rightData.length < this.config.minSamplesLeaf) {
        continue;
      }

      const gain = this.calculateSplitGain(data, target, leftData, rightData);
      
      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = {
          feature,
          threshold,
          operator: 'lte',
          gain,
          leftData,
          rightData,
          leftCount: leftData.length,
          rightCount: rightData.length
        };
      }
    }

    return bestSplit;
  }

  /**
   * Finds the best split for a discrete feature
   */
  private findBestDiscreteSplit(
    data: TrainingData[],
    target: string,
    feature: string
  ): CARTSplitInfo | null {
    const uniqueValues = [...new Set(data.map(row => row[feature]))];
    
    if (uniqueValues.length <= 1) return null;

    const targetValues = data.map(row => row[target]);
    const parentImpurity = this.calculateImpurity(targetValues);
    
    let bestGain = -Infinity;
    let bestSplit: CARTSplitInfo | null = null;

    // For discrete features, we create a binary split by grouping values
    const valueGroups = this.generateValueGroups(uniqueValues);
    
    for (const group of valueGroups) {
      const leftData = data.filter(row => group.includes(row[feature]));
      const rightData = data.filter(row => !group.includes(row[feature]));
      
      if (leftData.length < this.config.minSamplesLeaf || rightData.length < this.config.minSamplesLeaf) {
        continue;
      }

      const gain = this.calculateSplitGain(data, target, leftData, rightData);
      
      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = {
          feature,
          threshold: undefined,
          operator: 'eq',
          gain,
          leftData,
          rightData,
          leftCount: leftData.length,
          rightCount: rightData.length
        };
      }
    }

    return bestSplit;
  }

  /**
   * Calculates the gain of a split
   */
  private calculateSplitGain(
    parentData: TrainingData[],
    target: string,
    leftData: TrainingData[],
    rightData: TrainingData[]
  ): number {
    const parentTargetValues = parentData.map(row => row[target]);
    const leftTargetValues = leftData.map(row => row[target]);
    const rightTargetValues = rightData.map(row => row[target]);

    const parentImpurity = this.calculateImpurity(parentTargetValues);
    const leftImpurity = this.calculateImpurity(leftTargetValues);
    const rightImpurity = this.calculateImpurity(rightTargetValues);

    const leftWeight = leftData.length / parentData.length;
    const rightWeight = rightData.length / parentData.length;

    const weightedImpurity = leftWeight * leftImpurity + rightWeight * rightImpurity;
    return parentImpurity - weightedImpurity;
  }

  /**
   * Calculates impurity based on the configured criterion
   */
  private calculateImpurity(values: any[]): number {
    switch (this.config.criterion) {
      case 'gini':
        return this.calculateGiniImpurity(values);
      case 'entropy':
        return this.calculateEntropy(values);
      case 'mse':
        return this.calculateMSE(values);
      case 'mae':
        return this.calculateMAE(values);
      default:
        return this.calculateGiniImpurity(values);
    }
  }

  /**
   * Calculates Gini impurity
   */
  private calculateGiniImpurity(values: any[]): number {
    if (values.length === 0) return 0;

    const valueCounts = this.countValueOccurrences(values);
    const total = values.length;
    let gini = 1;

    for (const count of Object.values(valueCounts)) {
      const probability = count / total;
      gini -= probability * probability;
    }

    return gini;
  }

  /**
   * Calculates entropy
   */
  private calculateEntropy(values: any[]): number {
    if (values.length === 0) return 0;

    const valueCounts = this.countValueOccurrences(values);
    const total = values.length;
    let entropy = 0;

    for (const count of Object.values(valueCounts)) {
      const probability = count / total;
      if (probability > 0) {
        entropy -= probability * log2(probability);
      }
    }

    return entropy;
  }

  /**
   * Calculates Mean Squared Error (for regression)
   */
  private calculateMSE(values: number[]): number {
    if (values.length === 0) return 0;

    const numericValues = values.map(v => Number(v)).filter(n => !isNaN(n));
    if (numericValues.length === 0) return 0;

    const mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
    const mse = numericValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / numericValues.length;
    
    return mse;
  }

  /**
   * Calculates Mean Absolute Error (for regression)
   */
  private calculateMAE(values: number[]): number {
    if (values.length === 0) return 0;

    const numericValues = values.map(v => Number(v)).filter(n => !isNaN(n));
    if (numericValues.length === 0) return 0;

    const mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
    const mae = numericValues.reduce((sum, val) => sum + Math.abs(val - mean), 0) / numericValues.length;
    
    return mae;
  }

  /**
   * Generates possible value groups for discrete feature splitting
   */
  private generateValueGroups(uniqueValues: any[]): any[][] {
    if (uniqueValues.length <= 2) {
      return [uniqueValues.slice(0, 1)];
    }

    // For simplicity, create binary splits by taking the first half of values
    const mid = Math.floor(uniqueValues.length / 2);
    return [uniqueValues.slice(0, mid)];
  }

  /**
   * Creates a leaf node
   */
  private createLeafNode(value: any, sampleSize: number): TreeNode {
    return {
      type: NODE_TYPES.RESULT,
      val: value,
      name: String(value),
      alias: String(value) + randomUUID(),
      sampleSize
    };
  }

  /**
   * Checks if all values in an array are the same
   */
  private allSameValue(values: any[]): boolean {
    if (values.length <= 1) return true;
    const first = values[0];
    return values.every(val => _.isEqual(val, first));
  }

  /**
   * Gets the most common value in an array
   */
  private getMostCommonValue(values: any[]): any {
    if (values.length === 0) return null;

    const valueCounts = this.countValueOccurrences(values);
    let maxCount = 0;
    let mostCommon = values[0];

    for (const [valueStr, count] of Object.entries(valueCounts)) {
      if (count > maxCount) {
        maxCount = count;
        // Find the original value that matches this string representation
        mostCommon = values.find(v => String(v) === valueStr);
      }
    }

    return mostCommon;
  }

  /**
   * Gets the appropriate leaf value based on the criterion
   */
  private getLeafValue(values: any[]): any {
    if (values.length === 0) return null;
    
    // For regression tasks (MSE, MAE), use mean
    if (this.config.criterion === 'mse' || this.config.criterion === 'mae') {
      const numericValues = values.map(v => Number(v)).filter(n => !isNaN(n));
      if (numericValues.length > 0) {
        return numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
      }
    }
    
    // For classification tasks (gini, entropy), use most common value
    return this.getMostCommonValue(values);
  }

  /**
   * Counts occurrences of each value in an array
   */
  private countValueOccurrences(values: any[]): { [key: string]: number } {
    const counts: { [key: string]: number } = {};
    values.forEach(value => {
      const key = String(value);
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }
}

/**
 * Convenience function to create a CART tree
 */
export function createCARTTree(
  data: TrainingData[],
  target: string,
  features: string[],
  featureTypes: { [feature: string]: 'discrete' | 'continuous' },
  config?: Partial<CARTConfig>
): TreeNode {
  const cart = new CARTAlgorithm(config);
  return cart.createTree(data, target, features, featureTypes);
}
