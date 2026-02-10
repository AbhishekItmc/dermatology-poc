// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock WebGL context
const mockWebGLContext = {
  canvas: {},
  drawingBufferWidth: 800,
  drawingBufferHeight: 600,
  getParameter: jest.fn((param) => {
    // Return appropriate values for common parameters
    if (param === 0x8B4C) return 16; // MAX_VERTEX_ATTRIBS
    if (param === 0x8869) return 16; // MAX_TEXTURE_IMAGE_UNITS
    if (param === 0x8DFD) return 2048; // MAX_TEXTURE_SIZE
    return 0;
  }),
  getExtension: jest.fn(() => ({})),
  getShaderPrecisionFormat: jest.fn(() => ({
    precision: 23,
    rangeMin: 127,
    rangeMax: 127
  })),
  createShader: jest.fn(() => ({})),
  shaderSource: jest.fn(),
  compileShader: jest.fn(),
  getShaderParameter: jest.fn(() => true),
  getShaderInfoLog: jest.fn(() => ''),
  createProgram: jest.fn(() => ({})),
  attachShader: jest.fn(),
  linkProgram: jest.fn(),
  getProgramParameter: jest.fn(() => true),
  getProgramInfoLog: jest.fn(() => ''),
  useProgram: jest.fn(),
  createBuffer: jest.fn(() => ({})),
  bindBuffer: jest.fn(),
  bufferData: jest.fn(),
  enableVertexAttribArray: jest.fn(),
  vertexAttribPointer: jest.fn(),
  createTexture: jest.fn(() => ({})),
  bindTexture: jest.fn(),
  texImage2D: jest.fn(),
  texParameteri: jest.fn(),
  generateMipmap: jest.fn(),
  clear: jest.fn(),
  clearColor: jest.fn(),
  clearDepth: jest.fn(),
  enable: jest.fn(),
  disable: jest.fn(),
  depthFunc: jest.fn(),
  viewport: jest.fn(),
  drawArrays: jest.fn(),
  drawElements: jest.fn(),
  getAttribLocation: jest.fn(() => 0),
  getUniformLocation: jest.fn(() => ({})),
  uniform1f: jest.fn(),
  uniform1i: jest.fn(),
  uniform2f: jest.fn(),
  uniform3f: jest.fn(),
  uniform4f: jest.fn(),
  uniformMatrix3fv: jest.fn(),
  uniformMatrix4fv: jest.fn(),
  createFramebuffer: jest.fn(() => ({})),
  bindFramebuffer: jest.fn(),
  framebufferTexture2D: jest.fn(),
  checkFramebufferStatus: jest.fn(() => 0x8CD5), // FRAMEBUFFER_COMPLETE
  deleteBuffer: jest.fn(),
  deleteFramebuffer: jest.fn(),
  deleteProgram: jest.fn(),
  deleteShader: jest.fn(),
  deleteTexture: jest.fn(),
};

HTMLCanvasElement.prototype.getContext = jest.fn((contextType) => {
  if (contextType === 'webgl' || contextType === 'webgl2' || contextType === 'experimental-webgl') {
    return mockWebGLContext;
  }
  return null;
}) as any;

// Mock window.devicePixelRatio
Object.defineProperty(window, 'devicePixelRatio', {
  writable: true,
  configurable: true,
  value: 1,
});
