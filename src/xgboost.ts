/**
 * XGBoost Algorithm
 * @module XGBoost
 */

import _ from 'lodash';
import DecisionTree from './decision-tree.js';
import { 
  TreeNode, 
  DecisionTreeData, 
  TrainingData, 
  XGBoostConfig, 
  XGBoostData,
  BoostingHistory,
  NODE_TYPES 
} from './shared/types.js';
import { 
  SeededRandom 
} from './shared/utils.js';
import { LossFunctionFactory } from './shared/loss-functions.js';
import { 
  createWeightedTree, 
  createWeightedSample, 
  calculateBaseScore 
} from './shared/gradient-boosting.js';
import { DataTypeDetector, detectDataTypes, recommendAlgorithm } from './shared/data-type-detection.js';
import { globalCache, getCachedPrediction, setCachedPrediction } from './shared/caching-system.js';
import { processDataOptimized, OptimizedDataset } from './shared/memory-optimization.js';

/**
 * XGBoost class implementing gradient boosting with decision trees
 * Now supports both discrete and continuous variables with automatic algorithm selection
 */
class XGBoost {
  public static readonly NODE_TYPES = NODE_TYPES;
  
  private trees: DecisionTree[] = [];
  private data: any[] = [];
  private target!: string;
  private features!: string[];
  private config: XGBoostConfig;
  private baseScore: number = 0;
  private bestIteration: number = 0;
  private boostingHistory: BoostingHistory = {
    trainLoss: [],
    validationLoss: [],
    iterations: []
  };
  private featureTypes: Map<string, 'discrete' | 'continuous'> = new Map();
  private algorithm: 'id3' | 'cart' | 'hybrid' = 'auto';
  private optimizedDataset?: OptimizedDataset;
  private dataTypeDetector: DataTypeDetector;

  constructor(...args: any[]) {
    const numArgs = args.length;
    
    // Default configuration
    this.config = {
      nEstimators: 100,
      learningRate: 0.1,
      maxDepth: 6,
      minChildWeight: 1,
      subsample: 1,
      colsampleByTree: 1,
      regAlpha: 0,
      regLambda: 1,
      objective: 'regression',
      earlyStoppingRounds: undefined,
      randomState: undefined,
      validationFraction: 0.2,
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
        const instance = new XGBoost(target, features);
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
      const instance = new XGBoost(target, features, config);
      instance.train(data);
      return instance;
    }
    else {
      throw new Error('Invalid arguments passed to constructor. Check documentation on usage');
    }
  }

  /**
   * Trains the XGBoost model with provided data
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
    this.boostingHistory = {
      trainLoss: [],
      validationLoss: [],
      iterations: []
    };

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
    const objective = this.config.objective || 'regression';
    const learningRate = this.config.learningRate || 0.1;

    // Calculate base score
    this.baseScore = calculateBaseScore(data, this.target, objective);

    // Split data for validation if early stopping is enabled
    let trainData = data;
    let validationData: TrainingData[] = [];
    
    if (this.config.earlyStoppingRounds && this.config.validationFraction) {
      const validationSize = Math.floor(data.length * this.config.validationFraction);
      const shuffledIndices = Array.from({ length: data.length }, (_, i) => i);
      
      // Shuffle indices
      for (let i = shuffledIndices.length - 1; i > 0; i--) {
        const j = random.nextInt(i + 1);
        [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
      }
      
      const validationIndices = shuffledIndices.slice(0, validationSize);
      const trainIndices = shuffledIndices.slice(validationSize);
      
      validationData = validationIndices.map(i => data[i]);
      trainData = trainIndices.map(i => data[i]);
    }

    // Initialize predictions
    let predictions = new Array(trainData.length).fill(this.baseScore);
    let validationPredictions = new Array(validationData.length).fill(this.baseScore);

    // Get loss function
    const LossFunction = LossFunctionFactory.create(objective);

    // Boosting iterations
    let bestValidationLoss = Infinity;
    let noImprovementCount = 0;

    for (let i = 0; i < nEstimators; i++) {
      // Calculate gradients and hessians
      const targetValues = trainData.map(sample => sample[this.target]);
      let gradient: number[];
      let hessian: number[];
      
      if (objective === 'multiclass') {
        // For multiclass, we need to handle it differently
        gradient = new Array(predictions.length).fill(0);
        hessian = new Array(predictions.length).fill(1);
      } else {
        const result = LossFunction.calculateGradientsAndHessians(predictions, targetValues);
        gradient = result.gradient;
        hessian = result.hessian;
      }

      // Create weighted sample
      const weightedSample = createWeightedSample(trainData, this.config, random);
      weightedSample.gradients = gradient;
      weightedSample.hessians = hessian;

      // Build tree with continuous variable support
      const tree = createWeightedTree(
        weightedSample.data,
        this.target,
        this.features,
        weightedSample.weights,
        weightedSample.gradients,
        weightedSample.hessians,
        this.config
      );

      // Create DecisionTree instance with continuous variable support
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
        minSamplesSplit: 2,
        maxDepth: this.config.maxDepth
      };

      const treeData: DecisionTreeData = {
        model: tree,
        data: weightedSample.data,
        target: this.target,
        features: this.features,
        featureTypes: Object.fromEntries(this.featureTypes),
        algorithm: this.algorithm === 'hybrid' ? 'auto' : this.algorithm,
        config: treeConfig
      };
      
      const decisionTree = new DecisionTree(treeData);
      this.trees.push(decisionTree);

      // Update predictions
      for (let j = 0; j < trainData.length; j++) {
        const treePrediction = decisionTree.predict(trainData[j]);
        predictions[j] += learningRate * treePrediction;
      }

      // Update validation predictions
      for (let j = 0; j < validationData.length; j++) {
        const treePrediction = decisionTree.predict(validationData[j]);
        validationPredictions[j] += learningRate * treePrediction;
      }

      // Calculate losses
      const trainLoss = LossFunction.calculateLoss(predictions, targetValues);
      this.boostingHistory.trainLoss.push(trainLoss);
      this.boostingHistory.iterations.push(i + 1);

      if (validationData.length > 0) {
        const validationTargetValues = validationData.map(sample => sample[this.target]);
        const validationLoss = LossFunction.calculateLoss(validationPredictions, validationTargetValues);
        this.boostingHistory.validationLoss.push(validationLoss);

        // Early stopping check
        if (this.config.earlyStoppingRounds) {
          if (validationLoss < bestValidationLoss) {
            bestValidationLoss = validationLoss;
            this.bestIteration = i + 1;
            noImprovementCount = 0;
          } else {
            noImprovementCount++;
            if (noImprovementCount >= this.config.earlyStoppingRounds) {
              break;
            }
          }
        }
      }
    }

    // If no early stopping, best iteration is the last one
    if (!this.config.earlyStoppingRounds) {
      this.bestIteration = this.trees.length;
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
   * Predicts class/value for a given sample
   * @param sample - Sample data to predict
   * @returns Predicted value
   */
  predict(sample: TrainingData): any {
    if (this.trees.length === 0) {
      throw new Error('XGBoost has not been trained yet. Call train() first.');
    }

    if (!sample || typeof sample !== 'object' || Array.isArray(sample)) {
      throw new Error('Sample must be an object');
    }

    // Check cache first if enabled
    if (this.config.cachingEnabled) {
      const modelId = this.getModelId();
      const cachedPrediction = getCachedPrediction(sample, modelId);
      if (cachedPrediction !== null) {
        return cachedPrediction;
      }
    }

    let prediction = this.baseScore;
    const learningRate = this.config.learningRate || 0.1;

    for (let i = 0; i < this.bestIteration; i++) {
      const treePrediction = this.trees[i].predict(sample);
      prediction += learningRate * treePrediction;
    }

    // Apply objective-specific transformation
    const objective = this.config.objective || 'regression';
    if (objective === 'binary') {
      // Convert to probability using sigmoid
      const clampedPrediction = Math.max(-500, Math.min(500, prediction));
      const probability = 1 / (1 + Math.exp(-clampedPrediction));
      prediction = probability > 0.5 ? true : false;
    }

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
    return `xgb_${this.algorithm}_${this.target}_${this.features.join('_')}_${this.trees.length}`;
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
      
      // Handle different data types
      if (typeof actual === 'boolean') {
        pred = Boolean(pred);
      } else if (typeof actual === 'number') {
        pred = Number(pred);
      }
      
      if (_.isEqual(pred, actual)) {
        correct++;
      }
    });

    return correct / total;
  }

  /**
   * Imports a previously saved model
   * @param json - JSON representation of the model
   */
  import(json: XGBoostData): void {
    const {trees, target, features, config, data, baseScore, bestIteration, boostingHistory} = json;

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
    if (baseScore === undefined || typeof baseScore !== 'number') {
      throw new Error('Invalid model: baseScore property is required and must be a number');
    }
    if (bestIteration === undefined || typeof bestIteration !== 'number') {
      throw new Error('Invalid model: bestIteration property is required and must be a number');
    }
    if (!boostingHistory || typeof boostingHistory !== 'object') {
      throw new Error('Invalid model: boostingHistory property is required and must be an object');
    }

    this.trees = trees.map(treeData => {
      const tree = new DecisionTree(treeData);
      return tree;
    });
    
    this.data = data;
    this.target = target;
    this.features = features;
    this.config = config;
    this.baseScore = baseScore || 0;
    this.bestIteration = bestIteration || trees.length;
    this.boostingHistory = boostingHistory || {
      trainLoss: [],
      validationLoss: [],
      iterations: []
    };
  }

  /**
   * Returns JSON representation of trained model
   * @returns JSON object containing model data
   */
  toJSON(): XGBoostData {
    const {data, target, features, config} = this;
    const trees = this.trees.map(tree => tree.toJSON());

    return {
      trees, 
      data, 
      target, 
      features, 
      config,
      baseScore: this.baseScore,
      bestIteration: this.bestIteration,
      boostingHistory: this.boostingHistory
    };
  }

  /**
   * Gets feature importance scores
   * @returns Object with feature names as keys and importance scores as values
   */
  getFeatureImportance(): { [feature: string]: number } {
    if (this.trees.length === 0) {
      throw new Error('XGBoost has not been trained yet. Call train() first.');
    }

    const importance: { [feature: string]: number } = {};
    
    // Initialize all features with 0 importance
    this.features.forEach(feature => {
      importance[feature] = 0;
    });

    // Sum up importance from all trees
    for (let i = 0; i < this.bestIteration; i++) {
      const tree = this.trees[i];
      const treeJson = tree.toJSON();
      this.calculateTreeImportance(treeJson.model, importance);
    }

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
   * Gets the boosting history
   * @returns Boosting history with losses and iterations
   */
  getBoostingHistory(): BoostingHistory {
    return { ...this.boostingHistory };
  }

  /**
   * Gets the best iteration number
   * @returns Best iteration number
   */
  getBestIteration(): number {
    return this.bestIteration;
  }

  /**
   * Gets the number of trees in the model
   * @returns Number of trees
   */
  getTreeCount(): number {
    return this.trees.length;
  }

  /**
   * Gets the configuration used for this model
   * @returns Configuration object
   */
  getConfig(): XGBoostConfig {
    return { ...this.config };
  }

  /**
   * Gets the algorithm used by this model
   * @returns Algorithm name
   */
  getAlgorithm(): 'id3' | 'cart' | 'hybrid' {
    return this.algorithm;
  }

  /**
   * Gets the feature types detected for this model
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

// Export the XGBoost class
export default XGBoost;
