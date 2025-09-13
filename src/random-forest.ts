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

/**
 * Random Forest class implementing ensemble learning with multiple Decision Trees
 */
class RandomForest {
  public static readonly NODE_TYPES = NODE_TYPES;
  
  private trees: DecisionTree[] = [];
  private data: any[] = [];
  private target!: string;
  private features!: string[];
  private config: RandomForestConfig;

  constructor(...args: any[]) {
    const numArgs = args.length;
    
    // Default configuration
    this.config = {
      nEstimators: 100,
      maxFeatures: 'sqrt',
      bootstrap: true,
      randomState: undefined,
      maxDepth: undefined,
      minSamplesSplit: 2
    };
    
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

      // Create and train decision tree
      const tree = new DecisionTree(this.target, selectedFeatures);
      tree.train(trainingData);
      
      // Store the bootstrap sample data in the tree for testing purposes
      const treeJson = tree.toJSON();
      treeJson.data = trainingData;
      const treeWithData = new DecisionTree(treeJson);
      
      this.trees.push(treeWithData);
    }
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

    const predictions = this.trees.map(tree => tree.predict(sample));
    return majorityVote(predictions);
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
    this.config = config;
  }

  /**
   * Returns JSON representation of trained model
   * @returns JSON object containing model data
   */
  toJSON(): RandomForestData {
    const {data, target, features, config} = this;
    const trees = this.trees.map(tree => tree.toJSON());

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
}

// Export the RandomForest class
export default RandomForest;
