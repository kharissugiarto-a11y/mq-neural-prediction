#include <Arduino.h>
#include <EEPROM.h>
#include <math.h>
#include <stddef.h>

/*
  MQ Neural Dataset Collector - Arduino Nano ATmega328P

  A0 = MQ-6
  A1 = MQ-2
  A2 = MQ-135
  A3 = MQ-3
  A4 = MQ-131

  Output hanya ADC, R0, dan Rs/R0. PPM tidak dihitung di firmware ini.

  MQ131_IS_LOW:
    1 = MQ131-L / konsentrasi rendah
    0 = MQ131-H / konsentrasi tinggi

  Perintah serial:
    C + Enter = kalibrasi udara bersih dan simpan R0 ke EEPROM
    E + Enter = hapus kalibrasi EEPROM
*/

#define MQ131_IS_LOW 1

const unsigned long SERIAL_BAUD = 115200UL;
const unsigned long SAMPLE_INTERVAL_MS = 1000UL;
const uint8_t ADC_SAMPLE_COUNT = 16;
const uint8_t CALIBRATION_SAMPLE_COUNT = 60;
const unsigned int CALIBRATION_SAMPLE_DELAY_MS = 250;

enum SensorIndex {
  IDX_MQ6 = 0,
  IDX_MQ2,
  IDX_MQ135,
  IDX_MQ3,
  IDX_MQ131,
  SENSOR_COUNT
};

const uint8_t SENSOR_PINS[SENSOR_COUNT] = {A0, A1, A2, A3, A4};

// Nilai Rs/R0 udara bersih dari kurva referensi yang dipakai proyek awal.
// MQ131 menggunakan R0 = Rs udara bersih sehingga rasionya 1.
const float CLEAN_AIR_RATIO[SENSOR_COUNT] = {10.0, 9.83, 3.60, 60.0, 1.0};

const uint32_t EEPROM_MAGIC = 0x4D514E31UL;  // "MQN1"
const uint8_t EEPROM_VERSION = 1;

struct CalibrationRecord {
  uint32_t magic;
  uint8_t version;
  uint8_t mq131Low;
  float r0[SENSOR_COUNT];
  uint16_t crc;
};

float r0[SENSOR_COUNT] = {NAN, NAN, NAN, NAN, NAN};
uint16_t adcValue[SENSOR_COUNT] = {0, 0, 0, 0, 0};
bool calibrationValid = false;
unsigned long lastSampleMs = 0;

uint16_t crc16(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; bit++) {
      crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
    }
  }
  return crc;
}

uint16_t readAdcAverage(uint8_t pin) {
  analogRead(pin);
  delayMicroseconds(250);

  uint32_t total = 0;
  for (uint8_t i = 0; i < ADC_SAMPLE_COUNT; i++) {
    total += analogRead(pin);
    delayMicroseconds(120);
  }
  return (uint16_t)((total + ADC_SAMPLE_COUNT / 2) / ADC_SAMPLE_COUNT);
}

void readAllAdc() {
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    adcValue[i] = readAdcAverage(SENSOR_PINS[i]);
  }
}

// Rs/RL = (1023 / ADC) - 1. RL hilang ketika dibagi dengan R0/RL.
float resistanceFactor(uint16_t adc) {
  if (adc == 0 || adc >= 1023) return NAN;
  return (1023.0 / (float)adc) - 1.0;
}

float ratioFor(uint8_t index) {
  if (!calibrationValid || index >= SENSOR_COUNT || !isfinite(r0[index]) || r0[index] <= 0.0) {
    return NAN;
  }
  const float rs = resistanceFactor(adcValue[index]);
  if (!isfinite(rs) || rs <= 0.0) return NAN;
  return rs / r0[index];
}

bool loadCalibration() {
  CalibrationRecord record;
  EEPROM.get(0, record);

  const uint16_t expected = crc16((const uint8_t *)&record, offsetof(CalibrationRecord, crc));
  if (record.magic != EEPROM_MAGIC ||
      record.version != EEPROM_VERSION ||
      record.mq131Low != (MQ131_IS_LOW ? 1 : 0) ||
      record.crc != expected) {
    return false;
  }

  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    if (!isfinite(record.r0[i]) || record.r0[i] <= 0.0) return false;
    r0[i] = record.r0[i];
  }
  return true;
}

void saveCalibration() {
  CalibrationRecord record = {};
  record.magic = EEPROM_MAGIC;
  record.version = EEPROM_VERSION;
  record.mq131Low = MQ131_IS_LOW ? 1 : 0;
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) record.r0[i] = r0[i];
  record.crc = crc16((const uint8_t *)&record, offsetof(CalibrationRecord, crc));
  EEPROM.put(0, record);
}

void eraseCalibration() {
  CalibrationRecord empty = {};
  EEPROM.put(0, empty);
  calibrationValid = false;
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) r0[i] = NAN;
  Serial.println(F("#CALIBRATION_ERASED"));
}

void calibrateCleanAir() {
  Serial.println(F("#CALIBRATION_STARTED Keep every sensor in clean, ventilated air."));
  float totalRs[SENSOR_COUNT] = {0, 0, 0, 0, 0};
  uint8_t validCount[SENSOR_COUNT] = {0, 0, 0, 0, 0};

  for (uint8_t sample = 0; sample < CALIBRATION_SAMPLE_COUNT; sample++) {
    readAllAdc();
    for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
      const float rs = resistanceFactor(adcValue[i]);
      if (isfinite(rs) && rs > 0.0) {
        totalRs[i] += rs;
        validCount[i]++;
      }
    }
    delay(CALIBRATION_SAMPLE_DELAY_MS);
  }

  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    if (validCount[i] < CALIBRATION_SAMPLE_COUNT / 2) {
      calibrationValid = false;
      Serial.print(F("#CALIBRATION_FAILED sensor_index="));
      Serial.println(i);
      return;
    }
    const float rsClean = totalRs[i] / validCount[i];
    r0[i] = rsClean / CLEAN_AIR_RATIO[i];
  }

  calibrationValid = true;
  saveCalibration();
  Serial.println(F("#CALIBRATION_SAVED"));
}

void printJsonFloat(float value, uint8_t decimals) {
  if (!isfinite(value)) Serial.print(F("null"));
  else Serial.print(value, decimals);
}

void printSensorObject(const float values[SENSOR_COUNT], uint8_t decimals) {
  Serial.print(F("{\"mq6\":"));
  printJsonFloat(values[IDX_MQ6], decimals);
  Serial.print(F(",\"mq2\":"));
  printJsonFloat(values[IDX_MQ2], decimals);
  Serial.print(F(",\"mq135\":"));
  printJsonFloat(values[IDX_MQ135], decimals);
  Serial.print(F(",\"mq3\":"));
  printJsonFloat(values[IDX_MQ3], decimals);
  Serial.print(F(",\"mq131\":"));
  printJsonFloat(values[IDX_MQ131], decimals);
  Serial.print('}');
}

void emitJson() {
  float ratio[SENSOR_COUNT];
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) ratio[i] = ratioFor(i);

#if MQ131_IS_LOW
  const char *mq131Model = "LOW";
#else
  const char *mq131Model = "HIGH";
#endif

  Serial.print(F("{\"schema\":1,\"ms\":"));
  Serial.print(millis());
  Serial.print(F(",\"calibrated\":"));
  Serial.print(calibrationValid ? F("true") : F("false"));
  Serial.print(F(",\"mq131_model\":\""));
  Serial.print(mq131Model);

  Serial.print(F("\",\"adc\":{\"mq6\":"));
  Serial.print(adcValue[IDX_MQ6]);
  Serial.print(F(",\"mq2\":"));
  Serial.print(adcValue[IDX_MQ2]);
  Serial.print(F(",\"mq135\":"));
  Serial.print(adcValue[IDX_MQ135]);
  Serial.print(F(",\"mq3\":"));
  Serial.print(adcValue[IDX_MQ3]);
  Serial.print(F(",\"mq131\":"));
  Serial.print(adcValue[IDX_MQ131]);

  Serial.print(F("},\"r0\":"));
  printSensorObject(r0, 6);
  Serial.print(F(",\"ratio\":"));
  printSensorObject(ratio, 6);
  Serial.println('}');
}

void handleSerialCommand() {
  while (Serial.available() > 0) {
    const char command = Serial.read();
    if (command == 'C' || command == 'c') calibrateCleanAir();
    else if (command == 'E' || command == 'e') eraseCalibration();
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  analogReference(DEFAULT);
  delay(1000);

  calibrationValid = loadCalibration();
  Serial.println(F("#READY MQ neural dataset collector"));
  if (calibrationValid) {
    Serial.println(F("#CALIBRATION_LOADED"));
  } else {
    Serial.println(F("#UNCALIBRATED Send C after initial preheat in clean air."));
  }
}

void loop() {
  handleSerialCommand();

  const unsigned long now = millis();
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;
    readAllAdc();
    emitJson();
  }
}

