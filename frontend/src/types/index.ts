/**
 * TypeScript type definitions
 */

export interface Patient {
  id: string;
  name: string;
  age?: number;
  skinType?: string;
}

export interface ImageSet {
  id: string;
  patientId: string;
  images: Image[];
  captureDate: string;
  angularCoverage: number;
  qualityScore: number;
}

export interface Image {
  id: string;
  filePath: string;
  resolution: [number, number];
  captureAngle: number;
  qualityMetrics: QualityMetrics;
}

export interface QualityMetrics {
  resolutionScore: number;
  lightingUniformity: number;
  focusScore: number;
  faceCoverage: number;
  overallScore: number;
  issues: string[];
}

export interface Analysis {
  id: string;
  patientId: string;
  imageSetId: string;
  analysisDate: string;
  pigmentationAreas: PigmentationArea[];
  wrinkles: Wrinkle[];
  mesh: Mesh;
  regionalDensityScores: Record<string, number>;
  skinTextureGrade: string;
  processingTimeSeconds: number;
}

export interface PigmentationArea {
  id: string;
  severity: 'Low' | 'Medium' | 'High';
  surfaceAreaMm2: number;
  density: number;
  colorDeviation: number;
  melaninIndex: number;
  centroid: [number, number];
}

export interface Wrinkle {
  id: string;
  type: 'micro' | 'regular';
  lengthMm: number;
  depthMm: number;
  widthMm: number;
  severity: string;
  region: string;
}

export interface Mesh {
  vertices: number[][];
  faces: number[][];
  normals: number[][];
  uvCoordinates: number[][];
  vertexColors: number[][];
  vertexLabels: number[];
  textureMap: string; // Base64 or URL
}

export interface Treatment {
  type: string;
  targetRegions: string[];
  parameters: Record<string, number>;
  expectedDurationDays: number;
}

export interface Simulation {
  id: string;
  analysisId: string;
  treatment: Treatment;
  predictedMesh: Mesh;
  confidenceScore: number;
  timeline: [number, Mesh][];
  recommendations: string[];
}

export interface AnalysisStatus {
  analysisId: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  message?: string;
}
