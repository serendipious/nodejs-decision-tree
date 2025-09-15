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
  // Continuous variable support
  splitThreshold?: number;
  splitOperator?: 'lte' | 'gt' | 'eq';
  statistics?: {
    mean?: number;
    variance?: number;
    sampleCount?: number;
  };
}

export interface DecisionTreeData {
  model: TreeNode;
  data: any[];
  target: string;
  features: string[];
  // Continuous variable support
  featureTypes?: { [feature: string]: 'discrete' | 'continuous' };
  algorithm?: 'id3' | 'cart' | 'auto';
  config?: DecisionTreeConfig;
}

export interface DecisionTreeConfig {
  algorithm?: 'auto' | 'id3' | 'cart';
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxDepth?: number;
  criterion?: 'gini' | 'entropy' | 'mse' | 'mae';
  continuousSplitting?: 'binary' | 'multiway';
  autoDetectTypes?: boolean;
  discreteThreshold?: number;
  continuousThreshold?: number;
  confidenceThreshold?: number;
  statisticalTests?: boolean;
  handleMissingValues?: boolean;
  numericOnlyContinuous?: boolean;
  cachingEnabled?: boolean;
  memoryOptimization?: boolean;
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
  // Continuous variable support
  algorithm?: 'auto' | 'id3' | 'cart' | 'hybrid';
  autoDetectTypes?: boolean;
  discreteThreshold?: number;
  continuousThreshold?: number;
  confidenceThreshold?: number;
  statisticalTests?: boolean;
  handleMissingValues?: boolean;
  numericOnlyContinuous?: boolean;
  cachingEnabled?: boolean;
  memoryOptimization?: boolean;
  criterion?: 'gini' | 'entropy' | 'mse' | 'mae';
  continuousSplitting?: 'binary' | 'multiway';
}

export interface RandomForestData {
  trees: DecisionTreeData[];
  target: string;
  features: string[];
  config: RandomForestConfig;
  data: any[];
}

export interface XGBoostConfig {
  nEstimators?: number;           // Number of boosting rounds
  learningRate?: number;          // Step size shrinkage (eta)
  maxDepth?: number;              // Maximum tree depth
  minChildWeight?: number;        // Minimum sum of instance weight in leaf
  subsample?: number;             // Fraction of samples for each tree
  colsampleByTree?: number;       // Fraction of features for each tree
  regAlpha?: number;              // L1 regularization (alpha)
  regLambda?: number;             // L2 regularization (lambda)
  objective?: 'regression' | 'binary' | 'multiclass'; // Loss function
  earlyStoppingRounds?: number;   // Early stopping patience
  randomState?: number;           // Random seed
  validationFraction?: number;    // Fraction for validation set
  // Continuous variable support
  algorithm?: 'auto' | 'id3' | 'cart' | 'hybrid';
  autoDetectTypes?: boolean;
  discreteThreshold?: number;
  continuousThreshold?: number;
  confidenceThreshold?: number;
  statisticalTests?: boolean;
  handleMissingValues?: boolean;
  numericOnlyContinuous?: boolean;
  cachingEnabled?: boolean;
  memoryOptimization?: boolean;
  criterion?: 'gini' | 'entropy' | 'mse' | 'mae';
  continuousSplitting?: 'binary' | 'multiway';
}

export interface XGBoostData {
  trees: DecisionTreeData[];
  target: string;
  features: string[];
  config: XGBoostConfig;
  data: any[];
  baseScore: number;
  bestIteration: number;
  boostingHistory: BoostingHistory;
}

export interface BoostingHistory {
  trainLoss: number[];
  validationLoss: number[];
  iterations: number[];
}

export interface GradientHessian {
  gradient: number[];
  hessian: number[];
}

export interface WeightedSample {
  data: any[];
  weights: number[];
  gradients: number[];
  hessians: number[];
}

// Node types constant
export const NODE_TYPES = {
  RESULT: 'result',
  FEATURE: 'feature',
  FEATURE_VALUE: 'feature_value'
} as const;

export type NodeType = typeof NODE_TYPES[keyof typeof NODE_TYPES];
