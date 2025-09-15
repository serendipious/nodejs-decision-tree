/**
 * Memory-Efficient Data Structures and Optimizations
 * Provides optimized data storage and processing for large datasets
 */

import { TrainingData } from './types.js';

export interface OptimizedDataset {
  continuousFeatures: Map<string, Float32Array>;
  discreteFeatures: Map<string, any[]>;
  targetValues: any[];
  indices: Map<string, Map<any, number[]>>;
  valueMapping: Map<string, Map<any, number>>;
  reverseMapping: Map<string, Map<number, any>>;
  featureTypes: Map<string, 'discrete' | 'continuous'>;
  sampleCount: number;
  featureCount: number;
}

export interface MemoryOptimizedDataProcessor {
  processData(data: TrainingData[], features: string[], target: string, featureTypes: Map<string, 'discrete' | 'continuous'>): OptimizedDataset;
  getFeatureValues(dataset: OptimizedDataset, feature: string): Float32Array | any[];
  getTargetValues(dataset: OptimizedDataset): any[];
  getSample(dataset: OptimizedDataset, index: number): TrainingData;
  getSubset(dataset: OptimizedDataset, indices: number[]): OptimizedDataset;
  clear(dataset: OptimizedDataset): void;
}

export class MemoryOptimizedProcessor implements MemoryOptimizedDataProcessor {
  private compressionEnabled: boolean;
  private maxMemoryUsage: number;

  constructor(config: { compressionEnabled?: boolean; maxMemoryUsage?: number } = {}) {
    this.compressionEnabled = config.compressionEnabled ?? true;
    this.maxMemoryUsage = config.maxMemoryUsage ?? 100 * 1024 * 1024; // 100MB
  }

  /**
   * Processes raw data into memory-optimized format
   */
  processData(
    data: TrainingData[], 
    features: string[], 
    target: string, 
    featureTypes: Map<string, 'discrete' | 'continuous'>
  ): OptimizedDataset {
    const sampleCount = data.length;
    const featureCount = features.length;
    
    const continuousFeatures = new Map<string, Float32Array>();
    const discreteFeatures = new Map<string, any[]>();
    const targetValues: any[] = new Array(sampleCount);
    const indices = new Map<string, Map<any, number[]>>();
    const valueMapping = new Map<string, Map<any, number>>();
    const reverseMapping = new Map<string, Map<number, any>>();

    // Initialize indices and mappings for each feature
    for (const feature of features) {
      indices.set(feature, new Map());
      valueMapping.set(feature, new Map());
      reverseMapping.set(feature, new Map());
    }

    // Process each feature
    for (const feature of features) {
      const featureType = featureTypes.get(feature) || 'discrete';
      
      if (featureType === 'continuous') {
        const values = new Float32Array(sampleCount);
        for (let i = 0; i < sampleCount; i++) {
          const value = Number(data[i][feature]);
          values[i] = isNaN(value) ? 0 : value;
        }
        continuousFeatures.set(feature, values);
      } else {
        const values: any[] = new Array(sampleCount);
        const uniqueValues = new Set<any>();
        
        for (let i = 0; i < sampleCount; i++) {
          const value = data[i][feature];
          values[i] = value;
          uniqueValues.add(value);
        }

        // Create value mapping for compression
        if (this.compressionEnabled && uniqueValues.size < sampleCount * 0.5) {
          const mapping = new Map<any, number>();
          const reverse = new Map<number, any>();
          let nextId = 0;

          for (const value of uniqueValues) {
            mapping.set(value, nextId);
            reverse.set(nextId, value);
            nextId++;
          }

          valueMapping.set(feature, mapping);
          reverseMapping.set(feature, reverse);
        }

        discreteFeatures.set(feature, values);
      }
    }

    // Process target values
    for (let i = 0; i < sampleCount; i++) {
      targetValues[i] = data[i][target];
    }

    // Build indices for fast lookups
    this.buildIndices(data, features, indices);

    return {
      continuousFeatures,
      discreteFeatures,
      targetValues,
      indices,
      valueMapping,
      reverseMapping,
      featureTypes,
      sampleCount,
      featureCount
    };
  }

  /**
   * Gets feature values in optimized format
   */
  getFeatureValues(dataset: OptimizedDataset, feature: string): Float32Array | any[] {
    if (dataset.continuousFeatures.has(feature)) {
      return dataset.continuousFeatures.get(feature)!;
    } else if (dataset.discreteFeatures.has(feature)) {
      return dataset.discreteFeatures.get(feature)!;
    } else {
      throw new Error(`Feature ${feature} not found in dataset`);
    }
  }

  /**
   * Gets target values
   */
  getTargetValues(dataset: OptimizedDataset): any[] {
    return dataset.targetValues;
  }

  /**
   * Gets a single sample from the dataset
   */
  getSample(dataset: OptimizedDataset, index: number): TrainingData {
    if (index < 0 || index >= dataset.sampleCount) {
      throw new Error(`Index ${index} out of range`);
    }

    const sample: TrainingData = {};

    // Add continuous features
    for (const [feature, values] of dataset.continuousFeatures.entries()) {
      sample[feature] = values[index];
    }

    // Add discrete features
    for (const [feature, values] of dataset.discreteFeatures.entries()) {
      sample[feature] = values[index];
    }

    return sample;
  }

  /**
   * Gets a subset of the dataset
   */
  getSubset(dataset: OptimizedDataset, indices: number[]): OptimizedDataset {
    const subsetCount = indices.length;
    const continuousFeatures = new Map<string, Float32Array>();
    const discreteFeatures = new Map<string, any[]>();
    const targetValues: any[] = new Array(subsetCount);

    // Process continuous features
    for (const [feature, values] of dataset.continuousFeatures.entries()) {
      const subsetValues = new Float32Array(subsetCount);
      for (let i = 0; i < subsetCount; i++) {
        subsetValues[i] = values[indices[i]];
      }
      continuousFeatures.set(feature, subsetValues);
    }

    // Process discrete features
    for (const [feature, values] of dataset.discreteFeatures.entries()) {
      const subsetValues: any[] = new Array(subsetCount);
      for (let i = 0; i < subsetCount; i++) {
        subsetValues[i] = values[indices[i]];
      }
      discreteFeatures.set(feature, subsetValues);
    }

    // Process target values
    for (let i = 0; i < subsetCount; i++) {
      targetValues[i] = dataset.targetValues[indices[i]];
    }

    return {
      continuousFeatures,
      discreteFeatures,
      targetValues,
      indices: new Map(), // Rebuild indices if needed
      valueMapping: dataset.valueMapping,
      reverseMapping: dataset.reverseMapping,
      featureTypes: dataset.featureTypes,
      sampleCount: subsetCount,
      featureCount: dataset.featureCount
    };
  }

  /**
   * Clears memory used by the dataset
   */
  clear(dataset: OptimizedDataset): void {
    dataset.continuousFeatures.clear();
    dataset.discreteFeatures.clear();
    dataset.targetValues.length = 0;
    dataset.indices.clear();
    dataset.valueMapping.clear();
    dataset.reverseMapping.clear();
    dataset.featureTypes.clear();
  }

  /**
   * Estimates memory usage of the dataset
   */
  estimateMemoryUsage(dataset: OptimizedDataset): number {
    let totalBytes = 0;

    // Continuous features (Float32Array)
    for (const values of dataset.continuousFeatures.values()) {
      totalBytes += values.length * 4; // 4 bytes per float32
    }

    // Discrete features (array of any)
    for (const values of dataset.discreteFeatures.values()) {
      totalBytes += values.length * 8; // Rough estimate
    }

    // Target values
    totalBytes += dataset.targetValues.length * 8;

    // Indices
    for (const featureIndices of dataset.indices.values()) {
      for (const indexArray of featureIndices.values()) {
        totalBytes += indexArray.length * 4; // 4 bytes per int32
      }
    }

    return totalBytes;
  }

  /**
   * Compresses discrete features using value mapping
   */
  compressDiscreteFeature(dataset: OptimizedDataset, feature: string): void {
    const values = dataset.discreteFeatures.get(feature);
    if (!values) return;

    const uniqueValues = new Set(values);
    if (uniqueValues.size >= values.length * 0.5) return; // Not worth compressing

    const mapping = new Map<any, number>();
    const reverse = new Map<number, any>();
    let nextId = 0;

    for (const value of uniqueValues) {
      mapping.set(value, nextId);
      reverse.set(nextId, value);
      nextId++;
    }

    dataset.valueMapping.set(feature, mapping);
    dataset.reverseMapping.set(feature, reverse);
  }

  /**
   * Decompresses discrete features
   */
  decompressDiscreteFeature(dataset: OptimizedDataset, feature: string): any[] {
    const values = dataset.discreteFeatures.get(feature);
    const mapping = dataset.reverseMapping.get(feature);
    
    if (!values || !mapping) return values || [];

    return values.map(id => mapping.get(id) || id);
  }

  private buildIndices(
    data: TrainingData[], 
    features: string[], 
    indices: Map<string, Map<any, number[]>>
  ): void {
    for (const feature of features) {
      const featureIndices = indices.get(feature)!;
      
      for (let i = 0; i < data.length; i++) {
        const value = data[i][feature];
        if (!featureIndices.has(value)) {
          featureIndices.set(value, []);
        }
        featureIndices.get(value)!.push(i);
      }
    }
  }
}

/**
 * Memory-efficient statistics calculator
 */
export class MemoryEfficientStatistics {
  private processor: MemoryOptimizedProcessor;

  constructor(processor: MemoryOptimizedProcessor) {
    this.processor = processor;
  }

  /**
   * Calculates statistics for a continuous feature efficiently
   */
  calculateContinuousStatistics(dataset: OptimizedDataset, feature: string): {
    mean: number;
    variance: number;
    std: number;
    min: number;
    max: number;
    quartiles: number[];
  } {
    const values = dataset.continuousFeatures.get(feature);
    if (!values) throw new Error(`Feature ${feature} not found or not continuous`);

    const sorted = new Float32Array(values).sort();
    const n = sorted.length;

    // Calculate basic statistics
    let sum = 0;
    let min = sorted[0];
    let max = sorted[n - 1];

    for (let i = 0; i < n; i++) {
      sum += sorted[i];
    }

    const mean = sum / n;

    // Calculate variance
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const diff = sorted[i] - mean;
      variance += diff * diff;
    }
    variance /= n;

    const std = Math.sqrt(variance);

    // Calculate quartiles
    const q1 = this.percentile(sorted, 25);
    const q2 = this.percentile(sorted, 50);
    const q3 = this.percentile(sorted, 75);

    return {
      mean,
      variance,
      std,
      min,
      max,
      quartiles: [q1, q2, q3]
    };
  }

  /**
   * Calculates statistics for a discrete feature efficiently
   */
  calculateDiscreteStatistics(dataset: OptimizedDataset, feature: string): {
    uniqueValues: any[];
    valueCounts: Map<any, number>;
    cardinality: number;
  } {
    const values = dataset.discreteFeatures.get(feature);
    if (!values) throw new Error(`Feature ${feature} not found or not discrete`);

    const valueCounts = new Map<any, number>();
    const uniqueValues = new Set<any>();

    for (const value of values) {
      uniqueValues.add(value);
      valueCounts.set(value, (valueCounts.get(value) || 0) + 1);
    }

    return {
      uniqueValues: Array.from(uniqueValues),
      valueCounts,
      cardinality: uniqueValues.size
    };
  }

  private percentile(sortedValues: Float32Array, p: number): number {
    const index = (p / 100) * (sortedValues.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    if (upper >= sortedValues.length) return sortedValues[sortedValues.length - 1];
    if (lower === upper) return sortedValues[lower];

    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  }
}

/**
 * Global memory-optimized processor instance
 */
export const globalMemoryProcessor = new MemoryOptimizedProcessor();

/**
 * Convenience functions
 */
export function processDataOptimized(
  data: TrainingData[], 
  features: string[], 
  target: string, 
  featureTypes: Map<string, 'discrete' | 'continuous'>
): OptimizedDataset {
  return globalMemoryProcessor.processData(data, features, target, featureTypes);
}

export function estimateMemoryUsage(dataset: OptimizedDataset): number {
  return globalMemoryProcessor.estimateMemoryUsage(dataset);
}
