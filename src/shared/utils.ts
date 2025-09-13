/**
 * Shared utility functions for Decision Tree and Random Forest
 */

import _ from 'lodash';

/**
 * Generates random UUID
 * @private
 */
export function randomUUID(): string {
  return "_r" + Math.random().toString(32).slice(2);
}

/**
 * Computes probability of a given value existing in a given list
 * @private
 */
export function prob(value: any, list: any[]): number {
  let occurrences = _.filter(list, function (element) {
    return element === value;
  });

  let numOccurrences = occurrences.length;
  let numElements = list.length;
  return numOccurrences / numElements;
}

/**
 * Computes Log with base-2
 * @private
 */
export function log2(n: number): number {
  return Math.log(n) / Math.log(2);
}

/**
 * Finds element with highest occurrence in a list
 * @private
 */
export function mostCommon(list: any[]): any {
  let elementFrequencyMap: { [key: string]: number } = {};
  let largestFrequency = -1;
  let mostCommonElement: any = null;

  list.forEach(function (element) {
    let elementFrequency = (elementFrequencyMap[element] || 0) + 1;
    elementFrequencyMap[element] = elementFrequency;

    if (largestFrequency < elementFrequency) {
      mostCommonElement = element;
      largestFrequency = elementFrequency;
    }
  });

  return mostCommonElement;
}

/**
 * Simple seeded random number generator for reproducible results
 * @private
 */
export class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  next(): number {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }

  nextInt(max: number): number {
    return Math.floor(this.next() * max);
  }
}

/**
 * Bootstrap sampling with replacement
 * @private
 */
export function bootstrapSample(data: any[], sampleSize: number, random: SeededRandom): any[] {
  if (data.length === 0) {
    throw new Error('Cannot create bootstrap sample from empty data');
  }
  
  const sample: any[] = [];
  for (let i = 0; i < sampleSize; i++) {
    const randomIndex = random.nextInt(data.length);
    sample.push(data[randomIndex]);
  }
  return sample;
}

/**
 * Random feature selection for Random Forest
 * @private
 */
export function selectRandomFeatures(
  features: string[], 
  maxFeatures: number | 'sqrt' | 'log2' | 'auto', 
  random: SeededRandom
): string[] {
  const totalFeatures = features.length;
  let numFeatures: number;

  switch (maxFeatures) {
    case 'sqrt':
      numFeatures = Math.floor(Math.sqrt(totalFeatures));
      break;
    case 'log2':
      numFeatures = Math.floor(Math.log2(totalFeatures));
      break;
    case 'auto':
      numFeatures = Math.floor(Math.sqrt(totalFeatures));
      break;
    default:
      numFeatures = Math.min(maxFeatures, totalFeatures);
  }

  // Ensure we have at least 1 feature
  numFeatures = Math.max(1, numFeatures);

  // Shuffle and select features
  const shuffledFeatures = [...features];
  for (let i = shuffledFeatures.length - 1; i > 0; i--) {
    const j = random.nextInt(i + 1);
    [shuffledFeatures[i], shuffledFeatures[j]] = [shuffledFeatures[j], shuffledFeatures[i]];
  }

  return shuffledFeatures.slice(0, numFeatures);
}

/**
 * Majority voting for ensemble predictions
 * @private
 */
export function majorityVote(predictions: any[]): any {
  const frequencyMap: { [key: string]: number } = {};
  
  predictions.forEach(prediction => {
    const key = String(prediction);
    frequencyMap[key] = (frequencyMap[key] || 0) + 1;
  });

  let maxCount = 0;
  let result: any = null;

  Object.entries(frequencyMap).forEach(([key, count]) => {
    if (count > maxCount) {
      maxCount = count;
      result = key;
    }
  });

  // Convert back to original type if possible
  const firstPrediction = predictions[0];
  if (typeof firstPrediction === 'boolean') {
    return result === 'true';
  } else if (typeof firstPrediction === 'number') {
    return Number(result);
  }
  
  return result;
}
