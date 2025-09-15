/**
 * Intelligent Caching System for Performance Optimization
 * Provides caching for statistics, split points, and computations
 */

import { TrainingData } from './types.js';

export interface CachedStatistics {
  mean: number;
  variance: number;
  std: number;
  min: number;
  max: number;
  quartiles: number[];
  uniqueValues: Set<any>;
  sortedValues: number[];
  lastUpdated: number;
  sampleCount: number;
}

export interface CachedSplitPoint {
  threshold: number;
  gain: number;
  leftCount: number;
  rightCount: number;
  lastUpdated: number;
}

export interface CachedNodeData {
  entropy: number;
  gini: number;
  sampleCount: number;
  targetDistribution: Map<any, number>;
  lastUpdated: number;
}

export interface CacheConfig {
  maxSize: number;
  ttl: number; // Time to live in milliseconds
  enableStatisticsCache: boolean;
  enableSplitPointCache: boolean;
  enableNodeCache: boolean;
  enablePredictionCache: boolean;
}

export class PerformanceCache {
  private statisticsCache = new Map<string, CachedStatistics>();
  private splitPointCache = new Map<string, CachedSplitPoint>();
  private nodeCache = new Map<string, CachedNodeData>();
  private predictionCache = new Map<string, any>();
  private config: CacheConfig;
  private accessCounts = new Map<string, number>();
  private lastAccess = new Map<string, number>();

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      maxSize: 1000,
      ttl: 300000, // 5 minutes
      enableStatisticsCache: true,
      enableSplitPointCache: true,
      enableNodeCache: true,
      enablePredictionCache: true,
      ...config
    };
  }

  /**
   * Gets cached statistics for a feature
   */
  getStatistics(feature: string, data: TrainingData[]): CachedStatistics | null {
    if (!this.config.enableStatisticsCache) return null;

    const key = this.generateStatisticsKey(feature, data);
    const cached = this.statisticsCache.get(key);

    if (cached && this.isValid(cached.lastUpdated)) {
      this.updateAccess(key);
      return cached;
    }

    if (cached) {
      this.statisticsCache.delete(key);
    }

    return null;
  }

  /**
   * Sets cached statistics for a feature
   */
  setStatistics(feature: string, data: TrainingData[], statistics: CachedStatistics): void {
    if (!this.config.enableStatisticsCache) return;

    const key = this.generateStatisticsKey(feature, data);
    this.setCacheEntry(key, statistics, this.statisticsCache);
  }

  /**
   * Gets cached split point for a feature
   */
  getSplitPoint(feature: string, data: TrainingData[]): CachedSplitPoint | null {
    if (!this.config.enableSplitPointCache) return null;

    const key = this.generateSplitPointKey(feature, data);
    const cached = this.splitPointCache.get(key);

    if (cached && this.isValid(cached.lastUpdated)) {
      this.updateAccess(key);
      return cached;
    }

    if (cached) {
      this.splitPointCache.delete(key);
    }

    return null;
  }

  /**
   * Sets cached split point for a feature
   */
  setSplitPoint(feature: string, data: TrainingData[], splitPoint: CachedSplitPoint): void {
    if (!this.config.enableSplitPointCache) return;

    const key = this.generateSplitPointKey(feature, data);
    this.setCacheEntry(key, splitPoint, this.splitPointCache);
  }

  /**
   * Gets cached node data
   */
  getNodeData(nodeId: string, data: TrainingData[]): CachedNodeData | null {
    if (!this.config.enableNodeCache) return null;

    const key = this.generateNodeKey(nodeId, data);
    const cached = this.nodeCache.get(key);

    if (cached && this.isValid(cached.lastUpdated)) {
      this.updateAccess(key);
      return cached;
    }

    if (cached) {
      this.nodeCache.delete(key);
    }

    return null;
  }

  /**
   * Sets cached node data
   */
  setNodeData(nodeId: string, data: TrainingData[], nodeData: CachedNodeData): void {
    if (!this.config.enableNodeCache) return;

    const key = this.generateNodeKey(nodeId, data);
    this.setCacheEntry(key, nodeData, this.nodeCache);
  }

  /**
   * Gets cached prediction
   */
  getPrediction(sample: TrainingData, modelId: string): any | null {
    if (!this.config.enablePredictionCache) return null;

    const key = this.generatePredictionKey(sample, modelId);
    const cached = this.predictionCache.get(key);

    if (cached && this.isValid(cached.lastUpdated)) {
      this.updateAccess(key);
      return cached;
    }

    if (cached) {
      this.predictionCache.delete(key);
    }

    return null;
  }

  /**
   * Sets cached prediction
   */
  setPrediction(sample: TrainingData, modelId: string, prediction: any): void {
    if (!this.config.enablePredictionCache) return;

    const key = this.generatePredictionKey(sample, modelId);
    this.setCacheEntry(key, prediction, this.predictionCache);
  }

  /**
   * Clears all caches
   */
  clear(): void {
    this.statisticsCache.clear();
    this.splitPointCache.clear();
    this.nodeCache.clear();
    this.predictionCache.clear();
    this.accessCounts.clear();
    this.lastAccess.clear();
  }

  /**
   * Clears expired entries
   */
  clearExpired(): void {
    const now = Date.now();
    
    this.clearExpiredFromCache(this.statisticsCache, now);
    this.clearExpiredFromCache(this.splitPointCache, now);
    this.clearExpiredFromCache(this.nodeCache, now);
    this.clearExpiredFromCache(this.predictionCache, now);
  }

  /**
   * Gets cache statistics
   */
  getCacheStats(): {
    statisticsCache: { size: number; hitRate: number };
    splitPointCache: { size: number; hitRate: number };
    nodeCache: { size: number; hitRate: number };
    predictionCache: { size: number; hitRate: number };
    totalSize: number;
    memoryUsage: number;
  } {
    const stats = {
      statisticsCache: this.getCacheStatsForMap(this.statisticsCache),
      splitPointCache: this.getCacheStatsForMap(this.splitPointCache),
      nodeCache: this.getCacheStatsForMap(this.nodeCache),
      predictionCache: this.getCacheStatsForMap(this.predictionCache),
      totalSize: 0,
      memoryUsage: 0
    };

    stats.totalSize = stats.statisticsCache.size + stats.splitPointCache.size + 
                     stats.nodeCache.size + stats.predictionCache.size;

    // Estimate memory usage (rough calculation)
    stats.memoryUsage = this.estimateMemoryUsage();

    return stats;
  }

  private generateStatisticsKey(feature: string, data: TrainingData[]): string {
    const dataHash = this.hashData(data);
    return `stats_${feature}_${dataHash}`;
  }

  private generateSplitPointKey(feature: string, data: TrainingData[]): string {
    const dataHash = this.hashData(data);
    return `split_${feature}_${dataHash}`;
  }

  private generateNodeKey(nodeId: string, data: TrainingData[]): string {
    const dataHash = this.hashData(data);
    return `node_${nodeId}_${dataHash}`;
  }

  private generatePredictionKey(sample: TrainingData, modelId: string): string {
    const sampleHash = this.hashObject(sample);
    return `pred_${modelId}_${sampleHash}`;
  }

  private hashData(data: TrainingData[]): string {
    // Simple hash based on data length and first few values
    if (data.length === 0) return 'empty';
    
    const sample = data[0];
    const keys = Object.keys(sample).sort();
    const hash = keys.map(key => `${key}:${sample[key]}`).join('|');
    return `${data.length}_${hash}`;
  }

  private hashObject(obj: any): string {
    return JSON.stringify(obj, Object.keys(obj).sort());
  }

  private isValid(timestamp: number): boolean {
    return Date.now() - timestamp < this.config.ttl;
  }

  private updateAccess(key: string): void {
    this.accessCounts.set(key, (this.accessCounts.get(key) || 0) + 1);
    this.lastAccess.set(key, Date.now());
  }

  private setCacheEntry<T>(key: string, value: T, cache: Map<string, T>): void {
    // Check if we need to evict entries
    if (cache.size >= this.config.maxSize) {
      this.evictLeastRecentlyUsed(cache);
    }

    cache.set(key, value);
    this.lastAccess.set(key, Date.now());
  }

  private evictLeastRecentlyUsed<T>(cache: Map<string, T>): void {
    let oldestKey = '';
    let oldestTime = Date.now();

    for (const [key, value] of cache.entries()) {
      const lastAccessTime = this.lastAccess.get(key) || 0;
      if (lastAccessTime < oldestTime) {
        oldestTime = lastAccessTime;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      cache.delete(oldestKey);
      this.lastAccess.delete(oldestKey);
      this.accessCounts.delete(oldestKey);
    }
  }

  private clearExpiredFromCache<T extends { lastUpdated: number }>(
    cache: Map<string, T>, 
    now: number
  ): void {
    for (const [key, value] of cache.entries()) {
      if (now - value.lastUpdated >= this.config.ttl) {
        cache.delete(key);
        this.lastAccess.delete(key);
        this.accessCounts.delete(key);
      }
    }
  }

  private getCacheStatsForMap<T extends { lastUpdated: number }>(
    cache: Map<string, T>
  ): { size: number; hitRate: number } {
    const size = cache.size;
    const totalAccesses = Array.from(this.accessCounts.values()).reduce((a, b) => a + b, 0);
    const hitRate = totalAccesses > 0 ? 
      Array.from(this.accessCounts.values()).reduce((a, b) => a + b, 0) / totalAccesses : 0;

    return { size, hitRate };
  }

  private estimateMemoryUsage(): number {
    // Rough estimation of memory usage in bytes
    let totalBytes = 0;

    // Statistics cache
    for (const [key, stats] of this.statisticsCache.entries()) {
      totalBytes += key.length * 2; // String length * 2 bytes per char
      totalBytes += 8 * 8; // 8 numbers * 8 bytes each
      totalBytes += stats.uniqueValues.size * 8; // Set entries
      totalBytes += stats.sortedValues.length * 8; // Array entries
    }

    // Split point cache
    for (const [key, split] of this.splitPointCache.entries()) {
      totalBytes += key.length * 2;
      totalBytes += 4 * 8; // 4 numbers * 8 bytes each
    }

    // Node cache
    for (const [key, node] of this.nodeCache.entries()) {
      totalBytes += key.length * 2;
      totalBytes += 3 * 8; // 3 numbers * 8 bytes each
      totalBytes += node.targetDistribution.size * 16; // Map entries
    }

    // Prediction cache
    for (const [key, pred] of this.predictionCache.entries()) {
      totalBytes += key.length * 2;
      totalBytes += JSON.stringify(pred).length * 2;
    }

    return totalBytes;
  }
}

/**
 * Global cache instance
 */
export const globalCache = new PerformanceCache();

/**
 * Convenience functions for global cache
 */
export function getCachedStatistics(feature: string, data: TrainingData[]): CachedStatistics | null {
  return globalCache.getStatistics(feature, data);
}

export function setCachedStatistics(feature: string, data: TrainingData[], statistics: CachedStatistics): void {
  globalCache.setStatistics(feature, data, statistics);
}

export function getCachedSplitPoint(feature: string, data: TrainingData[]): CachedSplitPoint | null {
  return globalCache.getSplitPoint(feature, data);
}

export function setCachedSplitPoint(feature: string, data: TrainingData[], splitPoint: CachedSplitPoint): void {
  globalCache.setSplitPoint(feature, data, splitPoint);
}

export function getCachedPrediction(sample: TrainingData, modelId: string): any | null {
  return globalCache.getPrediction(sample, modelId);
}

export function setCachedPrediction(sample: TrainingData, modelId: string, prediction: any): void {
  globalCache.setPrediction(sample, modelId, prediction);
}
