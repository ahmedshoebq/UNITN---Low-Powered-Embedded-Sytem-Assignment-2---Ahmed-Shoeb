# UNITN---Low-Powered-Embedded-Sytem-Assignment-2---Ahmed-Shoeb
# Keyword Spotting — Arduino Nano 33 BLE Sense Rev2

## Student
Name: Ahmed Shoeb Submission Date: 24-May-2026 Matricola: 265924

## Hardware
- Arduino Nano 33 BLE Sense Rev2
- Built-in MP34DT05 microphone — no external wiring needed
- Connected via USB only

## Sound Classes
- Clap
- Tap
- Snap
- Silence

## Full Pipeline
1. Microphone captures audio at 16kHz
2. 512 sample frames accumulated in circular buffer
3. MFCC extracted on Arduino using CMSIS DSP
4. 13 MFCC coefficients per frame
5. Features normalized using StandardScaler values
6. CNN model runs inference via TFLite
7. Predicted sound printed to Serial Monitor

## MFCC Pipeline (CMSIS DSP)
1. Audio sampling at 16kHz
2. Frame segmentation — 512 samples per frame
3. Hamming window — arm_mult_f32()
4. RFFT — arm_rfft_fast_f32()
5. Power spectrum — arm_cmplx_mag_squared_f32()
6. Mel filter bank — 26 filters from 0 to 8000 Hz
7. Log energy — log() applied to each mel band
8. DCT — manual implementation
9. 13 MFCC coefficients output

## Libraries Required
### Arduino
- PDM by Arduino — microphone input
- arm_math.h — CMSIS DSP for FFT and vector operations
- TensorFlowLite — on-device inference

### Python (Google Colab)
- numpy, pandas, sklearn, tensorflow, matplotlib, seaborn

## How to Collect Data
1. Open arduino/data_collection/data_collection.ino
2. Change LABEL to the sound name
3. Upload to Arduino Nano 33 BLE
4. Open Serial Monitor at 115200 baud
5. Make the sound repeatedly until DONE appears
6. Copy Serial Monitor output and save as sound_name.csv

## How to Train
1. Open training/keyword_training.ipynb in Google Colab
2. Upload all CSV files from data/ folder
3. Run all cells in order
4. keyword_model.h downloads automatically
5. Copy scaler values from Cell 6 output

## How to Run Inference
1. Place keyword_model.h in arduino/inference/ folder
2. Paste scaler values into keyword_inference.ino
3. Upload to Arduino Nano 33 BLE
4. Open Serial Monitor at 115200 baud
5. Make sounds near the board
6. Predictions appear in Serial Monitor

## Model Architecture
- Input: 13 MFCC features
- Conv1D: 32 filters, kernel 3, ReLU
- BatchNormalization
- MaxPooling1D
- Conv1D: 64 filters, kernel 3, ReLU
- BatchNormalization
- GlobalAveragePooling1D
- Dense: 64 neurons, ReLU
- Dropout: 0.3
- Output: 4 neurons, Softmax
- Deployed as TFLite on Arduino

## Results
- Confusion matrix available in training notebook
- Live predictions via Serial Monitor at 115200 baud
