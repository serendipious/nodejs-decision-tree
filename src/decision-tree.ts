import _ from 'lodash';
import { 
  TreeNode, 
  DecisionTreeData, 
  TrainingData, 
  NODE_TYPES,
  DecisionTreeConfig
} from './shared/types.js';
import { randomUUID, prob, log2, mostCommon } from './shared/utils.js';
import { createTree, entropy, gain, maxGain } from './shared/id3-algorithm.js';
import { createCARTTree, CARTConfig } from './shared/cart-algorithm.js';
import { DataTypeDetector, detectDataTypes, recommendAlgorithm } from './shared/data-type-detection.js';
import { globalCache, getCachedPrediction, setCachedPrediction } from './shared/caching-system.js';
import { processDataOptimized, OptimizedDataset } from './shared/memory-optimization.js';

/**
 * Decision Tree Algorithm
 * @module DecisionTree
 */

/**
 * Decision Tree class implementing ID3 and CART algorithms with continuous variable support
 */
class DecisionTree {
  public static readonly NODE_TYPES = NODE_TYPES;
  
  private model!: TreeNode;
  private data: any[] = [];
  private target!: string;
  private features!: string[];
  private config: DecisionTreeConfig;
  private featureTypes: Map<string, 'discrete' | 'continuous'> = new Map();
  private algorithm: 'id3' | 'cart' | 'auto' = 'auto';
  private optimizedDataset?: OptimizedDataset;
  private dataTypeDetector: DataTypeDetector;

  constructor(...args: any[]) {
    const numArgs = args.length;
    
    // Default configuration
    this.config = {
      algorithm: 'auto',
      minSamplesSplit: 2,
      minSamplesLeaf: 1,
      maxDepth: undefined,
      criterion: 'gini',
      continuousSplitting: 'binary',
      autoDetectTypes: true,
      discreteThreshold: 20,
      continuousThreshold: 20,
      confidenceThreshold: 0.7,
      statisticalTests: true,
      handleMissingValues: true,
      numericOnlyContinuous: true,
      cachingEnabled: true,
      memoryOptimization: true
    };

    this.dataTypeDetector = new DataTypeDetector({
      discreteThreshold: this.config.discreteThreshold,
      continuousThreshold: this.config.continuousThreshold,
      confidenceThreshold: this.config.confidenceThreshold,
      statisticalTests: this.config.statisticalTests,
      handleMissingValues: this.config.handleMissingValues,
      numericOnlyContinuous: this.config.numericOnlyContinuous
    });

    // Configuration validation will be called after merging user input
    
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
      // Check if third argument is an array (data) or object (config)
      if (Array.isArray(args[2])) {
        // [data, target, features] pattern
        const [data, target, features] = args;
        const instance = new DecisionTree(target, features);
        instance.train(data);
        return instance;
      } else {
        // [target, features, config] pattern
        const [target, features, config] = args;
        
        if (!target || typeof target !== 'string') {
          throw new Error('`target` argument is expected to be a String. Check documentation on usage');
        }
        if (!features || !Array.isArray(features)) {
          throw new Error('`features` argument is expected to be an Array<String>. Check documentation on usage');
        }
        if (config && typeof config === 'object') {
          this.config = { ...this.config, ...config };
        }

        this.target = target;
        this.features = features;
        
        // Validate configuration after merging user input
        this.validateConfig();
      }
    }
    else if (numArgs === 4) {
      const [data, target, features, config] = args;
      const instance = new DecisionTree(target, features, config);
      instance.train(data);
      return instance;
    }
    
    // Validate configuration for cases where it wasn't validated yet
    if (numArgs === 2) {
      this.validateConfig();
    }
    else if (numArgs !== 1 && numArgs !== 3 && numArgs !== 4) {
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

    this.data = data;

    // Detect data types if auto-detection is enabled
    if (this.config.autoDetectTypes) {
      this.detectDataTypes(data);
    }

    // Determine algorithm if auto mode
    if (this.config.algorithm === 'auto') {
      this.selectAlgorithm(data);
    } else {
      this.algorithm = this.config.algorithm as 'id3' | 'cart';
    }

    // Create optimized dataset if memory optimization is enabled
    if (this.config.memoryOptimization) {
      this.optimizedDataset = processDataOptimized(data, this.features, this.target, this.featureTypes);
    }

    // Train the model using the selected algorithm
    this.model = this.createTree(data);
  }

  /**
   * Validates the configuration
   */
  private validateConfig(): void {
    const validAlgorithms = ['auto', 'id3', 'cart'];
    if (this.config.algorithm && !validAlgorithms.includes(this.config.algorithm)) {
      throw new Error(`Invalid algorithm: ${this.config.algorithm}. Must be one of: ${validAlgorithms.join(', ')}`);
    }

    const validCriteria = ['gini', 'entropy', 'mse', 'mae'];
    if (this.config.criterion && !validCriteria.includes(this.config.criterion)) {
      throw new Error(`Invalid criterion: ${this.config.criterion}. Must be one of: ${validCriteria.join(', ')}`);
    }

    const validSplitting = ['binary', 'multiway'];
    if (this.config.continuousSplitting && !validSplitting.includes(this.config.continuousSplitting)) {
      throw new Error(`Invalid continuousSplitting: ${this.config.continuousSplitting}. Must be one of: ${validSplitting.join(', ')}`);
    }
  }

  /**
   * Detects data types for all features
   */
  private detectDataTypes(data: TrainingData[]): void {
    const featureAnalysis = this.dataTypeDetector.analyzeFeatures(data, this.features);
    
    for (const [feature, analysis] of Object.entries(featureAnalysis)) {
      this.featureTypes.set(feature, analysis.type);
    }
  }

  /**
   * Selects the best algorithm based on data characteristics
   */
  private selectAlgorithm(data: TrainingData[]): void {
    const recommendation = recommendAlgorithm(data, this.features, this.target, {
      discreteThreshold: this.config.discreteThreshold,
      continuousThreshold: this.config.continuousThreshold,
      confidenceThreshold: this.config.confidenceThreshold,
      statisticalTests: this.config.statisticalTests,
      handleMissingValues: this.config.handleMissingValues,
      numericOnlyContinuous: this.config.numericOnlyContinuous
    });

    this.algorithm = recommendation.algorithm === 'hybrid' ? 'cart' : recommendation.algorithm;
  }

  /**
   * Creates the tree using the selected algorithm
   */
  private createTree(data: TrainingData[]): TreeNode {
    if (this.algorithm === 'cart') {
      const cartConfig: CARTConfig = {
        minSamplesSplit: this.config.minSamplesSplit || 2,
        minSamplesLeaf: this.config.minSamplesLeaf || 1,
        maxDepth: this.config.maxDepth,
        criterion: this.config.criterion || 'gini',
        continuousSplitting: this.config.continuousSplitting || 'binary'
      };

      return createCARTTree(data, this.target, this.features, Object.fromEntries(this.featureTypes), cartConfig);
    } else {
      // Use ID3 algorithm
      return createTree(
        data, 
        this.target, 
        this.features,
        this.config.maxDepth,
        this.config.minSamplesSplit
      );
    }
  }

  /**
   * Predicts class for a given sample
   * @param sample - Sample data to predict
   * @returns Predicted class value
   */
  predict(sample: TrainingData): any {
    // Check cache first if enabled
    if (this.config.cachingEnabled) {
      const modelId = this.getModelId();
      const cachedPrediction = getCachedPrediction(sample, modelId);
      if (cachedPrediction !== null) {
        return cachedPrediction;
      }
    }

    let root = this.model;
    while (root.type !== NODE_TYPES.RESULT) {
      let attr = root.name;
      let sampleVal = sample[attr];
      let childNode: TreeNode | undefined;

      // Handle continuous variables with threshold-based splitting
      if (root.splitThreshold !== undefined && root.splitOperator) {
        const numericVal = Number(sampleVal);
        if (!isNaN(numericVal)) {
          if (root.splitOperator === 'lte' && numericVal <= root.splitThreshold) {
            childNode = root.vals![0]; // Left child (<= threshold)
          } else if (root.splitOperator === 'gt' && numericVal > root.splitThreshold) {
            childNode = root.vals![1]; // Right child (> threshold)
          }
        }
      } else {
        // Handle discrete variables with exact matching
        childNode = _.find(root.vals, function (node) {
          return node.name == sampleVal;
        });
      }

      // For CART trees, we need to traverse through intermediate nodes
      if (childNode && childNode.child) {
        root = childNode.child;
      } else if (childNode) {
        root = childNode;
      } else {
        // Fallback to first child if no match found
        if (root.vals && root.vals.length > 0) {
          root = root.vals[0].child || root.vals[0];
        } else {
          break;
        }
      }
    }

    const prediction = root.val;

    // Cache prediction if enabled
    if (this.config.cachingEnabled) {
      const modelId = this.getModelId();
      setCachedPrediction(sample, modelId, prediction);
    }

    return prediction;
  }

  /**
   * Gets a unique model identifier for caching
   */
  private getModelId(): string {
    return `${this.algorithm}_${this.target}_${this.features.join('_')}`;
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
    const {model, data, target, features, featureTypes, algorithm, config} = json;

    this.model = model;
    this.data = data;
    this.target = target;
    this.features = features;
    
    // Restore continuous variable support
    if (featureTypes) {
      this.featureTypes = new Map(Object.entries(featureTypes));
    }
    
    if (algorithm) {
      this.algorithm = algorithm;
    }
    
    if (config) {
      this.config = { ...this.config, ...config };
    }
  }

  /**
   * Returns JSON representation of trained model
   * @returns JSON object containing model data
   */
  toJSON(): DecisionTreeData {
    const {target, features} = this;
    const model = this.model;
    const featureTypes = Object.fromEntries(this.featureTypes);

    return {
      model, 
      data: [], // Don't store training data in exported model
      target, 
      features,
      featureTypes,
      algorithm: this.algorithm,
      config: this.config
    };
  }

  /**
   * Gets the algorithm used by this tree
   * @returns Algorithm name
   */
  getAlgorithm(): 'id3' | 'cart' | 'auto' {
    return this.algorithm;
  }

  /**
   * Gets the feature types detected for this tree
   * @returns Map of feature names to their types
   */
  getFeatureTypes(): { [feature: string]: 'discrete' | 'continuous' } {
    return Object.fromEntries(this.featureTypes);
  }

  /**
   * Gets the configuration used by this tree
   * @returns Configuration object
   */
  getConfig(): DecisionTreeConfig {
    return { ...this.config };
  }

  /**
   * Gets cache statistics if caching is enabled
   * @returns Cache statistics or null if caching is disabled
   */
  getCacheStats(): any | null {
    if (!this.config.cachingEnabled) return null;
    return globalCache.getCacheStats();
  }

  /**
   * Clears the prediction cache
   */
  clearCache(): void {
    if (this.config.cachingEnabled) {
      globalCache.clear();
    }
  }
}


// Export the DecisionTree class
export default DecisionTree;
