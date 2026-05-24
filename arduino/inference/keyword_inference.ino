#include <PDM.h>
#include <arm_math.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include "keyword_model.h"

// =============================================
const int NUM_FEATURES = 13;
const int NUM_CLASSES = 4;
const char* CLASS_NAMES[] = {"clap", "silence", "snap", "tap"};

const float SCALER_MEAN[13] = {
  -77.154928f, 5.493569f, 2.100413f, 2.963457f, 2.878046f, 2.072300f,
  1.830021f, 1.130178f, 1.355042f, 1.300416f, 0.991215f, 0.823830f, 0.711002f
};

const float SCALER_SCALE[13] = {
  21.870116f, 4.170815f, 2.258736f, 1.435492f, 1.637160f, 1.336571f,
  1.183367f, 0.888916f, 0.935136f, 0.848852f, 0.763377f, 0.736158f, 0.667156f
};
// =============================================

#define SAMPLE_RATE      16000
#define FRAME_SIZE       512
#define NUM_MFCC         13
#define NUM_MEL_FILTERS  26

arm_rfft_fast_instance_f32 fftInstance;

// ── Circular buffer (exact copy from data-collection sketch) ──
short  circBuffer[FRAME_SIZE * 2];
int    circWrite = 0;
int    circCount = 0;

short  tempBuf[256];
short  sampleBuffer[FRAME_SIZE];
float  frame[FRAME_SIZE];
float  windowedFrame[FRAME_SIZE];
float  fftOutput[FRAME_SIZE];
float  powerSpectrum[FRAME_SIZE / 2];
float  melEnergies[NUM_MEL_FILTERS];
float  logMelEnergies[NUM_MEL_FILTERS];
float  mfcc[NUM_MFCC];
float  hammingWindow[FRAME_SIZE];

volatile bool newSamples = false;

int   melFilterStart[NUM_MEL_FILTERS];
int   melFilterEnd[NUM_MEL_FILTERS];
float melFilterWeights[NUM_MEL_FILTERS][50];
int   melFilterLen[NUM_MEL_FILTERS];

const tflite::Model* tfl_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ── PDM callback — identical to data-collection sketch ──
void onPDMdata() {
  int bytes = PDM.available();
  int count = bytes / 2;
  PDM.read(tempBuf, bytes);
  for (int i = 0; i < count; i++) {
    circBuffer[circWrite] = tempBuf[i];
    circWrite = (circWrite + 1) % (FRAME_SIZE * 2);
    if (circCount < FRAME_SIZE * 2) circCount++;
  }
  newSamples = true;
}

float hzToMel(float hz) {
  return 2595.0f * log10(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
  return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
}

void buildMelFilterBank() {
  float melLow  = hzToMel(0.0f);
  float melHigh = hzToMel(SAMPLE_RATE / 2.0f);

  float melPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
    melPoints[i] = melLow + i * (melHigh - melLow) / (NUM_MEL_FILTERS + 1);

  float hzPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
    hzPoints[i] = melToHz(melPoints[i]);

  int fftBins[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
    fftBins[i] = (int)floor((FRAME_SIZE + 1) * hzPoints[i] / SAMPLE_RATE);

  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    melFilterStart[m] = fftBins[m];
    melFilterEnd[m]   = fftBins[m + 2];
    melFilterLen[m]   = 0;
    int idx = 0;
    for (int k = fftBins[m]; k < fftBins[m + 2]; k++) {
      if (k < fftBins[m + 1])
        melFilterWeights[m][idx] = (float)(k - fftBins[m]) /
                                   (fftBins[m + 1] - fftBins[m]);
      else
        melFilterWeights[m][idx] = (float)(fftBins[m + 2] - k) /
                                   (fftBins[m + 2] - fftBins[m + 1]);
      idx++;
      melFilterLen[m]++;
    }
  }
}

void buildHammingWindow() {
  for (int i = 0; i < FRAME_SIZE; i++)
    hammingWindow[i] = 0.54f - 0.46f * cos(2.0f * PI * i / (FRAME_SIZE - 1));
}

float safe_div(float a, float b) {
  if (b == 0.0f || isinf(b) || isnan(b)) return 0.0f;
  float r = a / b;
  return (isnan(r) || isinf(r)) ? 0.0f : r;
}

// ── computeMFCC — identical to data-collection sketch ──
void computeMFCC(float* result) {
  // Copy most recent FRAME_SIZE samples from circular buffer
  int readPos = (circWrite - FRAME_SIZE + FRAME_SIZE * 2) % (FRAME_SIZE * 2);
  for (int i = 0; i < FRAME_SIZE; i++)
    sampleBuffer[i] = circBuffer[(readPos + i) % (FRAME_SIZE * 2)];

  for (int i = 0; i < FRAME_SIZE; i++)
    frame[i] = (float)sampleBuffer[i] / 32768.0f;

  arm_mult_f32(frame, hammingWindow, windowedFrame, FRAME_SIZE);
  arm_rfft_fast_f32(&fftInstance, windowedFrame, fftOutput, 0);
  arm_cmplx_mag_squared_f32(fftOutput, powerSpectrum, FRAME_SIZE / 2);

  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    melEnergies[m] = 0.0f;
    int binIdx = melFilterStart[m];
    for (int k = 0; k < melFilterLen[m]; k++) {
      if (binIdx < FRAME_SIZE / 2)
        melEnergies[m] += melFilterWeights[m][k] * powerSpectrum[binIdx];
      binIdx++;
    }
    if (melEnergies[m] < 1e-10f) melEnergies[m] = 1e-10f;
  }

  for (int m = 0; m < NUM_MEL_FILTERS; m++)
    logMelEnergies[m] = log(melEnergies[m]);

  for (int n = 0; n < NUM_MFCC; n++) {
    result[n] = 0.0f;
    for (int m = 0; m < NUM_MEL_FILTERS; m++)
      result[n] += logMelEnergies[m] *
                   cos(PI * n * (m + 0.5f) / NUM_MEL_FILTERS);
    result[n] *= sqrt(2.0f / NUM_MEL_FILTERS);
  }
}

void softmax(float* arr, int n) {
  float mx = arr[0];
  for (int i = 1; i < n; i++) if (arr[i] > mx) mx = arr[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    arr[i] = exp(arr[i] - mx);
    sum += arr[i];
  }
  for (int i = 0; i < n; i++) arr[i] = safe_div(arr[i], sum);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  arm_rfft_fast_init_f32(&fftInstance, FRAME_SIZE);
  buildHammingWindow();
  buildMelFilterBank();

  PDM.onReceive(onPDMdata);
  PDM.setGain(20);

  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Microphone failed!");
    while (1);
  }

  tfl_model = tflite::GetModel(model_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Microphone OK!");
  Serial.println("Ready. Make a sound near the board.");
}

void loop() {
  if (!newSamples || circCount < FRAME_SIZE) return;
  newSamples = false;

  float features[NUM_FEATURES];
  computeMFCC(features);

  // Normalize
  for (int i = 0; i < NUM_FEATURES; i++) {
    float val = safe_div(features[i] - SCALER_MEAN[i], SCALER_SCALE[i]);
    input->data.f[i] = (isnan(val) || isinf(val)) ? 0.0f : val;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  float probs[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++)
    probs[i] = output->data.f[i];
  softmax(probs, NUM_CLASSES);

  int best = 0;
  float best_val = probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (probs[i] > best_val) {
      best_val = probs[i];
      best = i;
    }
  }

  // Print all probs every frame
  Serial.print("Probs -> ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASS_NAMES[i]);
    Serial.print(": ");
    Serial.print(probs[i] * 100, 1);
    Serial.print("%  ");
  }
  Serial.println();

  if (best_val > 0.5f) {
    Serial.print(">>> ");
    Serial.print(CLASS_NAMES[best]);
    Serial.print("  (");
    Serial.print(best_val * 100, 1);
    Serial.println("%)");
  }

  delay(100); // match data-collection sketch timing
}
