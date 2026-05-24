#include <PDM.h>

#define SAMPLE_RATE  16000
#define FRAME_SIZE   512

short sampleBuffer[FRAME_SIZE];
volatile int samplesRead = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  PDM.onReceive(onPDMdata);
  PDM.setGain(20);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Microphone failed!");
    while (1);
  }
  Serial.println("Microphone OK! Clap near the board.");
}

void loop() {
  if (samplesRead > 0) {
    int maxVal = 0;
    for (int i = 0; i < samplesRead; i++) {
      if (abs(sampleBuffer[i]) > maxVal)
        maxVal = abs(sampleBuffer[i]);
    }
    Serial.print("Max amplitude: ");
    Serial.println(maxVal);
    samplesRead = 0;
  }
}
