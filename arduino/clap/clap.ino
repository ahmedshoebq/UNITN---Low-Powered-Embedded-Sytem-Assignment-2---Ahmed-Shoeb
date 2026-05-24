#include <PDM.h>
#include <arm_math.h>

// ======================================
// CHANGE THIS BEFORE EACH SESSION:
// "clap"   "tap"   "snap"   "silence"
// ======================================
String LABEL = "clap";

#define SAMPLE_RATE      16000
#define FRAME_SIZE       512
#define NUM_MFCC         13
#define NUM_MEL_FILTERS  26
#define NUM_FRAMES       60

arm_rfft_fast_instance_f32 fftInstance;

// Circular buffer to accumulate samples
short  circBuffer[FRAME_SIZE * 2];
int    circWrite = 0;
int    circCount = 0;

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
volatile int  newSampleCount = 0;
short tempBuf[256];

int frameCount = 0;
bool done = false;

int   melFilterStart[NUM_MEL_FILTERS];
int   melFilterEnd[NUM_MEL_FILTERS];
float melFilterWeights[NUM_MEL_FILTERS][50];
int   melFilterLen[NUM_MEL_FILTERS];

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
    hammingWindow[i] = 0.54f - 0.46f *
                       cos(2.0f * PI * i / (FRAME_SIZE - 1));
}

void computeMFCC() {
  // Step 1 — Copy frame from circular buffer
  int readPos = (circWrite - FRAME_SIZE + FRAME_SIZE * 2) % (FRAME_SIZE * 2);
  for (int i = 0; i < FRAME_SIZE; i++) {
    sampleBuffer[i] = circBuffer[(readPos + i) % (FRAME_SIZE * 2)];
  }

  // Step 2 — Convert to float
  for (int i = 0; i < FRAME_SIZE; i++)
    frame[i] = (float)sampleBuffer[i] / 32768.0f;

  // Step 3 — Hamming window using CMSIS
  arm_mult_f32(frame, hammingWindow, windowedFrame, FRAME_SIZE);

  // Step 4 — RFFT using CMSIS
  arm_rfft_fast_f32(&fftInstance, windowedFrame, fftOutput, 0);

  // Step 5 — Power spectrum using CMSIS
  arm_cmplx_mag_squared_f32(fftOutput, powerSpectrum, FRAME_SIZE / 2);

  // Step 6 — Mel filter bank
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

  // Step 7 — Log energy
  for (int m = 0; m < NUM_MEL_FILTERS; m++)
    logMelEnergies[m] = log(melEnergies[m]);

  // Step 8 — DCT to get MFCC
  for (int n = 0; n < NUM_MFCC; n++) {
    mfcc[n] = 0.0f;
    for (int m = 0; m < NUM_MEL_FILTERS; m++)
      mfcc[n] += logMelEnergies[m] *
                 cos(PI * n * (m + 0.5f) / NUM_MEL_FILTERS);
    mfcc[n] *= sqrt(2.0f / NUM_MEL_FILTERS);
  }
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

  Serial.print("label");
  for (int i = 0; i < NUM_MFCC; i++) {
    Serial.print(",mfcc");
    Serial.print(i);
  }
  Serial.println();

  delay(3000);
  Serial.println("# Recording started!");
}

void loop() {
  if (done) return;

  // Wait until we have enough samples in circular buffer
  if (newSamples && circCount >= FRAME_SIZE) {
    newSamples = false;

    computeMFCC();

    Serial.print(LABEL);
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(",");
      Serial.print(mfcc[i], 6);
    }
    Serial.println();

    frameCount++;

    if (frameCount >= NUM_FRAMES) {
      for (int i = 0; i < 5; i++) {
        Serial.println("# DONE");
        delay(300);
      }
      done = true;
    }

    delay(100); // small pause between frames
  }
}
