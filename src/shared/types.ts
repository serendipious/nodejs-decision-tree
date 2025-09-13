/**
 * Shared type definitions for Decision Tree and Random Forest
 */

export interface TreeNode {
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

export interface DecisionTreeData {
  model: TreeNode;
  data: any[];
  target: string;
  features: string[];
}

export interface FeatureGain {
  gain: number;
  name: string;
}

export interface TrainingData {
  [key: string]: any;
}

export interface RandomForestConfig {
  nEstimators?: number;        // Number of trees (default: 100)
  maxFeatures?: number | 'sqrt' | 'log2' | 'auto'; // Features per split
  bootstrap?: boolean;         // Use bootstrap sampling (default: true)
  randomState?: number;        // Random seed for reproducibility
  maxDepth?: number;           // Maximum tree depth
  minSamplesSplit?: number;    // Minimum samples to split
}

export interface RandomForestData {
  trees: DecisionTreeData[];
  target: string;
  features: string[];
  config: RandomForestConfig;
  data: any[];
}

// Node types constant
export const NODE_TYPES = {
  RESULT: 'result',
  FEATURE: 'feature',
  FEATURE_VALUE: 'feature_value'
} as const;

export type NodeType = typeof NODE_TYPES[keyof typeof NODE_TYPES];
