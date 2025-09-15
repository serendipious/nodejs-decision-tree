/**
 * Automatic Data Type Detection System
 * Determines if features and targets are discrete or continuous
 */

import _ from 'lodash';

export interface DataTypeInfo {
  type: 'discrete' | 'continuous';
  confidence: number;
  uniqueValues?: any[];
  statistics?: {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    quartiles?: number[];
    skewness?: number;
    kurtosis?: number;
  };
  cardinality?: number;
  missingValues?: number;
  missingPercentage?: number;
}

export interface FeatureAnalysis {
  [featureName: string]: DataTypeInfo;
}

export interface DataTypeDetectionConfig {
  discreteThreshold: number;        // Max unique values for discrete (default: 20)
  continuousThreshold: number;      // Min unique values for continuous (default: 20)
  confidenceThreshold: number;      // Min confidence for automatic detection (default: 0.7)
  statisticalTests: boolean;        // Use statistical tests for validation (default: true)
  handleMissingValues: boolean;     // Handle missing values in analysis (default: true)
  numericOnlyContinuous: boolean;   // Only consider numeric values as continuous (default: true)
}

export class DataTypeDetector {
  private config: DataTypeDetectionConfig;

  constructor(config: Partial<DataTypeDetectionConfig> = {}) {
    this.config = {
      discreteThreshold: 20,
      continuousThreshold: 20,
      confidenceThreshold: 0.7,
      statisticalTests: true,
      handleMissingValues: true,
      numericOnlyContinuous: true,
      ...config
    };
  }

  /**
   * Analyzes all features in a dataset to determine their data types
   * @param data - Training data array
   * @param features - Array of feature names to analyze
   * @returns Feature analysis with data type information
   */
  analyzeFeatures(data: any[], features: string[]): FeatureAnalysis {
    const analysis: FeatureAnalysis = {};

    for (const feature of features) {
      analysis[feature] = this.analyzeFeature(data, feature);
    }

    return analysis;
  }

  /**
   * Analyzes a single feature to determine its data type
   * @param data - Training data array
   * @param feature - Feature name to analyze
   * @returns Data type information for the feature
   */
  analyzeFeature(data: any[], feature: string): DataTypeInfo {
    const values = this.extractFeatureValues(data, feature);
    
    if (values.length === 0) {
      return {
        type: 'discrete',
        confidence: 0,
        uniqueValues: [],
        cardinality: 0,
        missingValues: data.length,
        missingPercentage: 100
      };
    }

    const uniqueValues = [...new Set(values)];
    const cardinality = uniqueValues.length;
    const missingValues = data.length - values.length;
    const missingPercentage = (missingValues / data.length) * 100;

    // Check if all values are numeric
    const numericValues = this.getNumericValues(values);
    const isNumeric = numericValues.length === values.length;

    // Basic heuristics
    const isDiscreteByCardinality = cardinality <= this.config.discreteThreshold;
    const isContinuousByCardinality = cardinality >= this.config.continuousThreshold;
    const isBoolean = cardinality === 2 && uniqueValues.every(v => typeof v === 'boolean');
    const isString = values.every(v => typeof v === 'string');
    const isNumericOnly = isNumeric && this.config.numericOnlyContinuous;

    let type: 'discrete' | 'continuous';
    let confidence: number;

    // Boolean values are always discrete
    if (isBoolean) {
      type = 'discrete';
      confidence = 1.0;
    }
    // String values are always discrete
    else if (isString) {
      type = 'discrete';
      confidence = 1.0;
    }
    // High cardinality numeric values are likely continuous
    else if (isNumericOnly && isContinuousByCardinality) {
      type = 'continuous';
      confidence = this.calculateContinuousConfidence(numericValues, cardinality);
    }
    // Low cardinality values are likely discrete
    else if (isDiscreteByCardinality) {
      type = 'discrete';
      confidence = this.calculateDiscreteConfidence(uniqueValues, cardinality);
    }
    // Medium cardinality - use statistical tests
    else if (isNumericOnly && this.config.statisticalTests) {
      const statisticalResult = this.performStatisticalTests(numericValues);
      type = statisticalResult.type;
      confidence = statisticalResult.confidence;
    }
    // Default to discrete for non-numeric or unclear cases
    else {
      type = 'discrete';
      confidence = 0.5;
    }

    const result: DataTypeInfo = {
      type,
      confidence,
      uniqueValues: uniqueValues.slice(0, 100), // Limit to first 100 unique values
      cardinality,
      missingValues,
      missingPercentage
    };

    // Add statistics for continuous features
    if (type === 'continuous' && isNumericOnly) {
      result.statistics = this.calculateStatistics(numericValues);
    }

    return result;
  }

  /**
   * Analyzes the target variable to determine if it's suitable for regression or classification
   * @param data - Training data array
   * @param target - Target variable name
   * @returns Data type information for the target
   */
  analyzeTarget(data: any[], target: string): DataTypeInfo {
    const targetInfo = this.analyzeFeature(data, target);
    
    // For targets, we also determine if it's suitable for regression
    if (targetInfo.type === 'continuous' && targetInfo.statistics) {
      const { mean, std, min, max } = targetInfo.statistics;
      
      if (mean !== undefined && std !== undefined && min !== undefined && max !== undefined) {
        const range = max - min;
        const coefficientOfVariation = std / mean;
        
        // High coefficient of variation suggests good regression target
        if (coefficientOfVariation > 0.1 && range > 0) {
          targetInfo.confidence = Math.min(targetInfo.confidence + 0.2, 1.0);
        }
      }
    }

    return targetInfo;
  }

  /**
   * Determines the best algorithm for a given dataset
   * @param featureAnalysis - Analysis of all features
   * @param targetAnalysis - Analysis of target variable
   * @returns Recommended algorithm and confidence
   */
  recommendAlgorithm(
    featureAnalysis: FeatureAnalysis, 
    targetAnalysis: DataTypeInfo
  ): { algorithm: 'id3' | 'cart' | 'hybrid'; confidence: number; reasoning: string[] } {
    const reasoning: string[] = [];
    let id3Score = 0;
    let cartScore = 0;
    let hybridScore = 0;

    const features = Object.keys(featureAnalysis);
    const discreteFeatures = features.filter(f => featureAnalysis[f].type === 'discrete');
    const continuousFeatures = features.filter(f => featureAnalysis[f].type === 'continuous');
    const totalFeatures = features.length;

    // Score based on feature types
    if (discreteFeatures.length === totalFeatures) {
      id3Score += 1.0;
      reasoning.push('All features are discrete - ID3 is optimal');
    } else if (continuousFeatures.length === totalFeatures) {
      cartScore += 1.0;
      reasoning.push('All features are continuous - CART is optimal');
    } else {
      hybridScore += 1.0;
      reasoning.push('Mixed feature types - Hybrid approach recommended');
    }

    // Score based on target type
    if (targetAnalysis.type === 'discrete') {
      id3Score += 0.3;
      cartScore += 0.2;
      reasoning.push('Discrete target - both ID3 and CART suitable');
    } else {
      cartScore += 0.5;
      reasoning.push('Continuous target - CART required for regression');
    }

    // Score based on dataset size and complexity
    const avgCardinality = features.reduce((sum, f) => sum + (featureAnalysis[f].cardinality || 0), 0) / totalFeatures;
    if (avgCardinality > 50) {
      cartScore += 0.2;
      reasoning.push('High cardinality features - CART handles better');
    }

    // Determine best algorithm
    const scores = { id3: id3Score, cart: cartScore, hybrid: hybridScore };
    const bestAlgorithm = Object.entries(scores).reduce((a, b) => scores[a[0] as keyof typeof scores] > scores[b[0] as keyof typeof scores] ? a : b)[0] as 'id3' | 'cart' | 'hybrid';
    const confidence = Math.min(scores[bestAlgorithm], 1.0);

    return {
      algorithm: bestAlgorithm,
      confidence,
      reasoning
    };
  }

  private extractFeatureValues(data: any[], feature: string): any[] {
    return data
      .map(row => row[feature])
      .filter(value => value !== null && value !== undefined && value !== '');
  }

  private getNumericValues(values: any[]): number[] {
    return values
      .map(v => Number(v))
      .filter(n => !isNaN(n) && isFinite(n));
  }

  private calculateContinuousConfidence(numericValues: number[], cardinality: number): number {
    let confidence = 0.5;

    // Higher cardinality increases confidence
    if (cardinality > this.config.continuousThreshold * 2) {
      confidence += 0.3;
    }

    // Check for continuous distribution patterns
    const sorted = numericValues.sort((a, b) => a - b);
    const range = sorted[sorted.length - 1] - sorted[0];
    const step = range / (cardinality - 1);
    const actualSteps = this.calculateStepSizes(sorted);
    const avgStep = actualSteps.reduce((a, b) => a + b, 0) / actualSteps.length;

    // If steps are relatively uniform, it's likely continuous
    if (Math.abs(avgStep - step) / step < 0.5) {
      confidence += 0.2;
    }

    return Math.min(confidence, 1.0);
  }

  private calculateDiscreteConfidence(uniqueValues: any[], cardinality: number): number {
    let confidence = 0.5;

    // Lower cardinality increases confidence for discrete
    if (cardinality <= this.config.discreteThreshold / 2) {
      confidence += 0.3;
    }

    // Check if values are evenly distributed (suggests discrete categories)
    const valueCounts = this.countValueOccurrences(uniqueValues);
    const counts = Object.values(valueCounts);
    const avgCount = counts.reduce((a, b) => a + b, 0) / counts.length;
    const variance = counts.reduce((sum, count) => sum + Math.pow(count - avgCount, 2), 0) / counts.length;
    const coefficientOfVariation = Math.sqrt(variance) / avgCount;

    // Low variation suggests discrete categories
    if (coefficientOfVariation < 0.5) {
      confidence += 0.2;
    }

    return Math.min(confidence, 1.0);
  }

  private performStatisticalTests(numericValues: number[]): { type: 'discrete' | 'continuous'; confidence: number } {
    // Simple normality test using skewness and kurtosis
    const statistics = this.calculateStatistics(numericValues);
    const { skewness = 0, kurtosis = 0 } = statistics;

    // Values closer to 0 for skewness and 3 for kurtosis suggest normal distribution (continuous)
    const skewnessScore = Math.max(0, 1 - Math.abs(skewness) / 2);
    const kurtosisScore = Math.max(0, 1 - Math.abs(kurtosis - 3) / 3);
    const normalityScore = (skewnessScore + kurtosisScore) / 2;

    if (normalityScore > 0.6) {
      return { type: 'continuous', confidence: normalityScore };
    } else {
      return { type: 'discrete', confidence: 1 - normalityScore };
    }
  }

  private calculateStatistics(values: number[]): {
    mean: number;
    std: number;
    min: number;
    max: number;
    quartiles: number[];
    skewness: number;
    kurtosis: number;
  } {
    const sorted = values.sort((a, b) => a - b);
    const n = sorted.length;
    
    const mean = sorted.reduce((a, b) => a + b, 0) / n;
    const variance = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    
    const min = sorted[0];
    const max = sorted[n - 1];
    
    // Quartiles
    const q1 = this.percentile(sorted, 25);
    const q2 = this.percentile(sorted, 50);
    const q3 = this.percentile(sorted, 75);
    
    // Skewness
    const skewness = n > 2 ? 
      (n / ((n - 1) * (n - 2))) * 
      sorted.reduce((sum, val) => sum + Math.pow((val - mean) / std, 3), 0) : 0;
    
    // Kurtosis
    const kurtosis = n > 3 ?
      ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) *
      sorted.reduce((sum, val) => sum + Math.pow((val - mean) / std, 4), 0) -
      (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3)) : 0;

    return {
      mean,
      std,
      min,
      max,
      quartiles: [q1, q2, q3],
      skewness,
      kurtosis
    };
  }

  private calculateStepSizes(sortedValues: number[]): number[] {
    const steps: number[] = [];
    for (let i = 1; i < sortedValues.length; i++) {
      steps.push(sortedValues[i] - sortedValues[i - 1]);
    }
    return steps;
  }

  private countValueOccurrences(values: any[]): { [key: string]: number } {
    const counts: { [key: string]: number } = {};
    values.forEach(value => {
      const key = String(value);
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }

  private percentile(sortedValues: number[], p: number): number {
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
 * Convenience function for quick data type detection
 */
export function detectDataTypes(
  data: any[], 
  features: string[], 
  config?: Partial<DataTypeDetectionConfig>
): FeatureAnalysis {
  const detector = new DataTypeDetector(config);
  return detector.analyzeFeatures(data, features);
}

/**
 * Convenience function for algorithm recommendation
 */
export function recommendAlgorithm(
  data: any[], 
  features: string[], 
  target: string, 
  config?: Partial<DataTypeDetectionConfig>
): { algorithm: 'id3' | 'cart' | 'hybrid'; confidence: number; reasoning: string[] } {
  const detector = new DataTypeDetector(config);
  const featureAnalysis = detector.analyzeFeatures(data, features);
  const targetAnalysis = detector.analyzeTarget(data, target);
  return detector.recommendAlgorithm(featureAnalysis, targetAnalysis);
}
