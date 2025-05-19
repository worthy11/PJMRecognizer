const fs = require("fs");
const path = require("path");

// Create model directory if it doesn't exist
const modelDir = path.join(__dirname, "public", "model");
if (!fs.existsSync(modelDir)) {
  fs.mkdirSync(modelDir, { recursive: true });
}

// Source directory for WASM files
const sourceDir = path.join(
  __dirname,
  "node_modules",
  "onnxruntime-web",
  "dist"
);

// Files to copy
const files = [
  "ort-wasm.wasm",
  "ort-wasm-simd.wasm",
  "ort-wasm-threaded.wasm",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm.min.js",
];

// Copy each file
files.forEach((file) => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(modelDir, file);

  if (fs.existsSync(sourcePath)) {
    fs.copyFileSync(sourcePath, destPath);
    console.log(`Copied ${file} to ${modelDir}`);
  } else {
    console.warn(`Warning: ${file} not found in ${sourceDir}`);
  }
});
