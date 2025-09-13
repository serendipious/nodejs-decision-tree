/**
 * Loss functions for XGBoost gradient boosting
 */

import { GradientHessian } from './types.js';

/**
 * Mean Squared Error loss function for regression
 */
export class MSELoss {
  static calculateGradient(prediction: number, actual: number): number {
    return prediction - actual;
  }

  static calculateHessian(prediction: number, actual: number): number {
    return 1;
  }

  static calculateLoss(predictions: number[], actuals: number[]): number {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i] - actuals[i];
      sum += diff * diff;
    }
    return sum / predictions.length;
  }

  static calculateGradientsAndHessians(
    predictions: number[], 
    actuals: number[]
  ): GradientHessian {
    const gradients: number[] = [];
    const hessians: number[] = [];

    for (let i = 0; i < predictions.length; i++) {
      gradients.push(this.calculateGradient(predictions[i], actuals[i]));
      hessians.push(this.calculateHessian(predictions[i], actuals[i]));
    }

    return { gradient: gradients, hessian: hessians };
  }
}

/**
 * Logistic loss function for binary classification
 */
export class LogisticLoss {
  static sigmoid(x: number): number {
    // Clamp x to prevent overflow
    const clampedX = Math.max(-500, Math.min(500, x));
    return 1 / (1 + Math.exp(-clampedX));
  }

  static calculateGradient(prediction: number, actual: number): number {
    const prob = this.sigmoid(prediction);
    return prob - actual;
  }

  static calculateHessian(prediction: number, actual: number): number {
    const prob = this.sigmoid(prediction);
    return prob * (1 - prob);
  }

  static calculateLoss(predictions: number[], actuals: number[]): number {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      const prob = this.sigmoid(predictions[i]);
      const actual = actuals[i];
      // Add small epsilon to prevent log(0)
      const epsilon = 1e-15;
      sum += actual * Math.log(prob + epsilon) + (1 - actual) * Math.log(1 - prob + epsilon);
    }
    return -sum / predictions.length;
  }

  static calculateGradientsAndHessians(
    predictions: number[], 
    actuals: number[]
  ): GradientHessian {
    const gradients: number[] = [];
    const hessians: number[] = [];

    for (let i = 0; i < predictions.length; i++) {
      gradients.push(this.calculateGradient(predictions[i], actuals[i]));
      hessians.push(this.calculateHessian(predictions[i], actuals[i]));
    }

    return { gradient: gradients, hessian: hessians };
  }
}

/**
 * Cross-entropy loss function for multiclass classification
 */
export class CrossEntropyLoss {
  static softmax(x: number[]): number[] {
    const max = Math.max(...x);
    const exp = x.map(val => Math.exp(val - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(val => val / sum);
  }

  static calculateGradient(predictions: number[], actual: number): number[] {
    const probs = this.softmax(predictions);
    const gradients = new Array(predictions.length).fill(0);
    gradients[actual] = 1;
    return gradients.map((grad, i) => grad - probs[i]);
  }

  static calculateHessian(predictions: number[], actual: number): number[][] {
    const probs = this.softmax(predictions);
    const hessians: number[][] = [];
    
    for (let i = 0; i < predictions.length; i++) {
      const row: number[] = [];
      for (let j = 0; j < predictions.length; j++) {
        if (i === j) {
          row.push(probs[i] * (1 - probs[i]));
        } else {
          row.push(-probs[i] * probs[j]);
        }
      }
      hessians.push(row);
    }
    
    return hessians;
  }

  static calculateLoss(predictions: number[], actuals: number[]): number {
    // For simplicity, treat as binary classification
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
      const actual = actuals[i];
      const prob = 1 / (1 + Math.exp(-predictions[i]));
      const epsilon = 1e-15;
      sum += actual * Math.log(prob + epsilon) + (1 - actual) * Math.log(1 - prob + epsilon);
    }
    return Math.max(0, -sum / predictions.length);
  }

  // Interface methods for compatibility
  static calculateGradientsAndHessians(predictions: number[], actuals: number[]): GradientHessian {
    // For simplicity, treat as binary classification
    const gradients: number[] = [];
    const hessians: number[] = [];
    
    for (let i = 0; i < predictions.length; i++) {
      const actual = actuals[i];
      const prob = 1 / (1 + Math.exp(-predictions[i]));
      gradients.push(prob - actual);
      hessians.push(prob * (1 - prob));
    }
    
    return { gradient: gradients, hessian: hessians };
  }
}

/**
 * Loss function interface
 */
export interface LossFunction {
  calculateGradientsAndHessians(predictions: number[], actuals: number[]): GradientHessian;
  calculateLoss(predictions: number[], actuals: number[]): number;
}

/**
 * Loss function factory
 */
export class LossFunctionFactory {
  static create(objective: 'regression' | 'binary' | 'multiclass'): LossFunction {
    switch (objective) {
      case 'regression':
        return MSELoss as LossFunction;
      case 'binary':
        return LogisticLoss as LossFunction;
      case 'multiclass':
        return CrossEntropyLoss as LossFunction;
      default:
        throw new Error(`Unsupported objective: ${objective}`);
    }
  }
}
