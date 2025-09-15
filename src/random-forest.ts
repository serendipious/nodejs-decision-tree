/**
 * Random Forest Algorithm
 * @module RandomForest
 */

import _ from 'lodash';
import DecisionTree from './decision-tree.js';
import { 
  TreeNode, 
  DecisionTreeData, 
  TrainingData, 
  RandomForestConfig, 
  RandomForestData,
  NODE_TYPES 
} from './shared/types.js';
import { 
  SeededRandom, 
  bootstrapSample, 
  selectRandomFeatures, 
  majorityVote 
} from './shared/utils.js';
import { createTree } from './shared/id3-algorithm.js';
import { DataTypeDetector, detectDataTypes, recommendAlgorithm } from './shared/data-type-detection.js';
import { globalCache, getCachedPrediction, setCachedPrediction } from './shared/caching-system.js';
import { processDataOptimized, OptimizedDataset } from './shared/memory-optimization.js';

/**
 * Random Forest class implementing ensemble learning with multiple Decision Trees
 * Now supports both discrete and continuous variables with automatic algorithm selection
 */
class RandomForest {
  public static readonly NODE_TYPES = NODE_TYPES;
  
  private trees: DecisionTree[] = [];
  private data: any[] = [];
  private target!: string;
  private features!: string[];
  private config: RandomForestConfig;
  private featureTypes: Map<string, 'discrete' | 'continuous'> = new Map();
  private algorithm: 'id3' | 'cart' | 'hybrid' | 'auto' = 'auto';
  private optimizedDataset?: OptimizedDataset;
  private dataTypeDetector: DataTypeDetector;

  constructor(...args: any[]) {
    const numArgs = args.length;
    
    // Default configuration
    this.config = {
      nEstimators: 100,
      maxFeatures: 'sqrt',
      bootstrap: true,
      randomState: undefined,
      maxDepth: undefined,
      minSamplesSplit: 2,
      // Continuous variable support
      algorithm: 'auto',
      autoDetectTypes: true,
      discreteThreshold: 20,
      continuousThreshold: 20,
      confidenceThreshold: 0.7,
      statisticalTests: true,
      handleMissingValues: true,
      numericOnlyContinuous: true,
      cachingEnabled: true,
      memoryOptimization: true,
      criterion: 'gini',
      continuousSplitting: 'binary'
    };

    this.dataTypeDetector = new DataTypeDetector({
      discreteThreshold: this.config.discreteThreshold,
      continuousThreshold: this.config.continuousThreshold,
      confidenceThreshold: this.config.confidenceThreshold,
      statisticalTests: this.config.statisticalTests,
      handleMissingValues: this.config.handleMissingValues,
      numericOnlyContinuous: this.config.numericOnlyContinuous
    });
    
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
        const instance = new RandomForest(target, features);
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
          // Update data type detector with new config
          this.dataTypeDetector = new DataTypeDetector({
            discreteThreshold: this.config.discreteThreshold,
            continuousThreshold: this.config.continuousThreshold,
            confidenceThreshold: this.config.confidenceThreshold,
            statisticalTests: this.config.statisticalTests,
            handleMissingValues: this.config.handleMissingValues,
            numericOnlyContinuous: this.config.numericOnlyContinuous
          });
        }

        this.target = target;
        this.features = features;
      }
    }
    else if (numArgs === 4) {
      const [data, target, features, config] = args;
      const instance = new RandomForest(target, features, config);
      instance.train(data);
      return instance;
    }
    else {
      throw new Error('Invalid arguments passed to constructor. Check documentation on usage');
    }
  }

  /**
   * Trains the random forest with provided data
   * @param data - Array of training data objects
   */
  train(data: TrainingData[]): void {
    if (!data || !Array.isArray(data)) {
      throw new Error('`data` argument is expected to be an Array<Object>. Check documentation on usage');
    }

    if (data.length === 0) {
      throw new Error('`data` argument is expected to be an Array<Object>. Check documentation on usage');
    }

    this.data = data;
    this.trees = [];

    // Detect data types if auto-detection is enabled
    if (this.config.autoDetectTypes) {
      this.detectDataTypes(data);
    }

    // Determine algorithm if auto mode
    if (this.config.algorithm === 'auto') {
      this.selectAlgorithm(data);
    } else {
      this.algorithm = this.config.algorithm as 'id3' | 'cart' | 'hybrid';
    }

    // Create optimized dataset if memory optimization is enabled
    if (this.config.memoryOptimization) {
      this.optimizedDataset = processDataOptimized(data, this.features, this.target, this.featureTypes);
    }
    
    const random = new SeededRandom(this.config.randomState || Math.floor(Math.random() * 1000000));
    const nEstimators = this.config.nEstimators !== undefined ? this.config.nEstimators : 100;
    const bootstrap = this.config.bootstrap !== false;
    const sampleSize = data.length;

    for (let i = 0; i < nEstimators; i++) {
      // Create bootstrap sample if enabled
      const trainingData = bootstrap ? bootstrapSample(data, sampleSize, random) : data;
      
      // Select random features for this tree
      const selectedFeatures = selectRandomFeatures(
        this.features, 
        this.config.maxFeatures || 'sqrt', 
        random
      );

      // Create decision tree with continuous variable support
      const treeConfig = {
        algorithm: this.algorithm === 'hybrid' ? 'auto' : this.algorithm,
        autoDetectTypes: false, // Already detected
        discreteThreshold: this.config.discreteThreshold,
        continuousThreshold: this.config.continuousThreshold,
        confidenceThreshold: this.config.confidenceThreshold,
        statisticalTests: this.config.statisticalTests,
        handleMissingValues: this.config.handleMissingValues,
        numericOnlyContinuous: this.config.numericOnlyContinuous,
        cachingEnabled: this.config.cachingEnabled,
        memoryOptimization: this.config.memoryOptimization,
        criterion: this.config.criterion,
        continuousSplitting: this.config.continuousSplitting,
        minSamplesSplit: this.config.minSamplesSplit,
        maxDepth: this.config.maxDepth
      };

      const tree = new DecisionTree(this.target, selectedFeatures, treeConfig);
      tree.train(trainingData);
      
      // Store the bootstrap sample data in the tree for testing purposes
      const treeJson = tree.toJSON();
      treeJson.data = trainingData;
      
      this.trees.push(tree);
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

    this.algorithm = recommendation.algorithm;
  }

  /**
   * Predicts class for a given sample using majority voting
   * @param sample - Sample data to predict
   * @returns Predicted class value
   */
  predict(sample: TrainingData): any {
    if (this.trees.length === 0) {
      throw new Error('Random Forest has not been trained yet. Call train() first.');
    }

    // Check cache first if enabled
    if (this.config.cachingEnabled) {
      const modelId = this.getModelId();
      const cachedPrediction = getCachedPrediction(sample, modelId);
      if (cachedPrediction !== null) {
        return cachedPrediction;
      }
    }

    const predictions = this.trees.map(tree => tree.predict(sample));
    const prediction = majorityVote(predictions);

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
    return `rf_${this.algorithm}_${this.target}_${this.features.join('_')}_${this.trees.length}`;
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
  import(json: RandomForestData): void {
    const {trees, target, features, config, data} = json;

    if (!trees || !Array.isArray(trees)) {
      throw new Error('Invalid model: trees property is required and must be an array');
    }
    if (!target || typeof target !== 'string') {
      throw new Error('Invalid model: target property is required and must be a string');
    }
    if (!features || !Array.isArray(features)) {
      throw new Error('Invalid model: features property is required and must be an array');
    }
    if (!config || typeof config !== 'object') {
      throw new Error('Invalid model: config property is required and must be an object');
    }
    if (!data || !Array.isArray(data)) {
      throw new Error('Invalid model: data property is required and must be an array');
    }

    this.trees = trees.map(treeData => {
      const tree = new DecisionTree(treeData);
      return tree;
    });
    
    this.data = data;
    this.target = target;
    this.features = features;
    this.config = { ...this.config, ...config };

    // Restore continuous variable support
    if (trees.length > 0) {
      const firstTree = this.trees[0];
      this.algorithm = firstTree.getAlgorithm() as 'id3' | 'cart' | 'hybrid';
      this.featureTypes = new Map(Object.entries(firstTree.getFeatureTypes()));
    }

    // Update data type detector with restored config
    this.dataTypeDetector = new DataTypeDetector({
      discreteThreshold: this.config.discreteThreshold,
      continuousThreshold: this.config.continuousThreshold,
      confidenceThreshold: this.config.confidenceThreshold,
      statisticalTests: this.config.statisticalTests,
      handleMissingValues: this.config.handleMissingValues,
      numericOnlyContinuous: this.config.numericOnlyContinuous
    });
  }

  /**
   * Returns JSON representation of trained model
   * @returns JSON object containing model data
   */
  toJSON(): RandomForestData {
    const {data, target, features, config} = this;
    const trees = this.trees.map(tree => {
      const treeJson = tree.toJSON();
      // Preserve bootstrap data for testing purposes
      if (this.config.bootstrap) {
        // Find the original bootstrap data from when the tree was trained
        // This is a simplified approach - in practice, you might want to store this differently
        treeJson.data = data; // Use original data for now
      }
      return treeJson;
    });

    return {trees, data, target, features, config};
  }

  /**
   * Gets feature importance scores based on information gain across all trees
   * @returns Object with feature names as keys and importance scores as values
   */
  getFeatureImportance(): { [feature: string]: number } {
    if (this.trees.length === 0) {
      throw new Error('Random Forest has not been trained yet. Call train() first.');
    }

    const importance: { [feature: string]: number } = {};
    
    // Initialize all features with 0 importance
    this.features.forEach(feature => {
      importance[feature] = 0;
    });

    // Sum up importance from all trees
    this.trees.forEach(tree => {
      const treeJson = tree.toJSON();
      this.calculateTreeImportance(treeJson.model, importance);
    });

    // Normalize by number of trees
    Object.keys(importance).forEach(feature => {
      importance[feature] /= this.trees.length;
    });

    return importance;
  }

  /**
   * Recursively calculates feature importance from a tree node
   * @private
   */
  private calculateTreeImportance(node: TreeNode, importance: { [feature: string]: number }): void {
    if (node.type === NODE_TYPES.FEATURE && node.gain && node.sampleSize) {
      const feature = node.name;
      const weightedGain = node.gain * node.sampleSize;
      importance[feature] = (importance[feature] || 0) + weightedGain;
    }

    if (node.vals) {
      node.vals.forEach(val => {
        if (val.child) {
          this.calculateTreeImportance(val.child, importance);
        }
      });
    }
  }

  /**
   * Gets the number of trees in the forest
   * @returns Number of trees
   */
  getTreeCount(): number {
    return this.trees.length;
  }

  /**
   * Gets the configuration used for this forest
   * @returns Configuration object
   */
  getConfig(): RandomForestConfig {
    return { ...this.config };
  }

  /**
   * Gets the algorithm used by this forest
   * @returns Algorithm name
   */
  getAlgorithm(): 'id3' | 'cart' | 'hybrid' | 'auto' {
    return this.algorithm;
  }

  /**
   * Gets the feature types detected for this forest
   * @returns Map of feature names to their types
   */
  getFeatureTypes(): { [feature: string]: 'discrete' | 'continuous' } {
    return Object.fromEntries(this.featureTypes);
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

// Export the RandomForest class
export default RandomForest;
