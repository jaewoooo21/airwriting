// AirWriting_ESP32_v3.ino
// WiFi & IMU UDP Streaming (9-Axis for S3, Dual I2C Buses)
// - SSID: 재우의 S25
// - Password: asdf750505*

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// WiFi credentials
const char *ssid = "재우의 S25";
const char *pass = "asdf750505*";

// UDP Configuration
const char *targetIP = "10.112.127.131"; // REPLACE WITH PC IP
const int targetPort = 12345;
const int localPort = 5555; // Port to receive data FROM python
WiFiUDP udp;

// OLED Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 *display;

// Hardware Pins
const int PEN_BTN_PIN = 15;
const int VIB_MOTOR_PIN = 4; // Haptic Feedback (Coin Motor)

// Debounce & Haptic State
uint8_t lastBtnState = 1; // The previous STABLE state
unsigned long vibTimer = 0;
bool isVibActive = false;

// Debounce variables
uint8_t stableBtnState = 1;       // Current STABLE button state
uint8_t lastUnstableBtnState = 1; // Previous RAW reading
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 100; // 30ms debounce delay

// I2C Bus Pins
const int I2C0_SDA = 21;
const int I2C0_SCL = 22;
const int I2C1_SDA = 32;
const int I2C1_SCL = 33;

TwoWire *I2C_MPU; // For MPU6050s
TwoWire *I2C_ICM; // For ICM20948

// IMU Addresses per Bus
// Bus 0 (MPU): S1 & S2
const int ADDR_S1_MPU = 0x68; // 전완근
const int ADDR_S2_MPU = 0x69; // 손등 (AD0 High)
// Bus 1 (ICM): S3
const int ADDR_S3_ICM = 0x68; // 손가락
const int ADDR_MAG = 0x0C;    // AK09916 내부 지자기센서

#pragma pack(push, 1)
// 6-axis data (24 bytes)
struct SensorData6 {
  float ax, ay, az;
  float gx, gy, gz;
};

// 9-axis data (36 bytes)
struct SensorData9 {
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
};

// Packet V3: S1(6-axis), S2(6-axis), S3(9-axis)
struct AirWritingPacketV3 {
  uint8_t header;     // 1B (0xAA)
  uint32_t timestamp; // 4B
  SensorData6 s1;     // 24B
  SensorData6 s2;     // 24B
  SensorData9 s3;     // 36B
  uint8_t button;     // 1B
  uint8_t checksum;   // 1B
  uint8_t footer;     // 1B (0x55)
};
#pragma pack(pop)

AirWritingPacketV3 packet;

// ICM-20948 뱅크 스위칭 (I2C_ICM 버스 전용)
void setICMBank(uint8_t bank) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x7F); // REG_BANK_SEL
  I2C_ICM->write(bank << 4);
  I2C_ICM->endTransmission(true);
}

void setupMPU6050(uint8_t addr) {
  // Wake up
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x6B); // PWR_MGMT_1
  I2C_MPU->write(0x01); // Auto-select clock
  I2C_MPU->endTransmission(true);

  // ACCEL_CONFIG (8g)
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1C);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);

  // GYRO_CONFIG (1000 dps)
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1B);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);
}

void setupICM20948() {
  setICMBank(0);

  // Wake up
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x6B); // PWR_MGMT_1
  I2C_ICM->write(0x01); // Auto-select clock
  I2C_ICM->endTransmission(true);
  delay(10);

  // Disable I2C Master, Enable Bypass
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x03); // USER_CTRL
  I2C_ICM->write(0x00);
  I2C_ICM->endTransmission(true);
  delay(10);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x0F); // INT_PIN_CFG
  I2C_ICM->write(0x02); // BYPASS_EN
  I2C_ICM->endTransmission(true);
  delay(10);

  // Configure Magnetometer (AK09916 via Bypass on I2C_ICM)
  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x31); // CNTL2
  I2C_ICM->write(0x08); // Cont. mode 2 (100Hz)
  I2C_ICM->endTransmission(true);
  delay(10);

  // ACCEL_CONFIG (Bank 2, 8g)
  setICMBank(2);
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x1C);
  I2C_ICM->write(0x10);
  I2C_ICM->endTransmission(true);

  // GYRO_CONFIG (Bank 2, 1000 dps)
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x01); // GYRO_CONFIG_1
  I2C_ICM->write(0x23); // 1000 dps, DLPF en
  I2C_ICM->endTransmission(true);

  // Return to Bank 0 for reading Data
  setICMBank(0);
}

void readMPU6050(uint8_t addr, SensorData6 &data) {
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x3B); // ACCEL_XOUT_H
  I2C_MPU->endTransmission(false);
  I2C_MPU->requestFrom((int)addr, 14, (int)true);

  if (I2C_MPU->available() == 14) {
    int16_t ax = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t ay = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t az = (I2C_MPU->read() << 8 | I2C_MPU->read());
    I2C_MPU->read();
    I2C_MPU->read(); // Skip
    int16_t gx = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gy = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gz = (I2C_MPU->read() << 8 | I2C_MPU->read());

    data.ax = ax * (9.81 / 4096.0);
    data.ay = ay * (9.81 / 4096.0);
    data.az = az * (9.81 / 4096.0);
    data.gx = gx * ((PI / 180.0) / 32.8);
    data.gy = gy * ((PI / 180.0) / 32.8);
    data.gz = gz * ((PI / 180.0) / 32.8);
  } else {
    data.ax = data.ay = data.az = 0;
    data.gx = data.gy = data.gz = 0;
  }
}

void readICM20948(SensorData9 &data) {
  // Read Accel & Gyro (Bank 0, 0x2D)
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x2D);
  I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_S3_ICM, 14, (int)true);

  if (I2C_ICM->available() == 14) {
    int16_t ax = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t ay = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t az = (I2C_ICM->read() << 8 | I2C_ICM->read());
    I2C_ICM->read();
    I2C_ICM->read(); // Skip
    int16_t gx = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gy = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gz = (I2C_ICM->read() << 8 | I2C_ICM->read());

    data.ax = ax * (9.81 / 4096.0);
    data.ay = ay * (9.81 / 4096.0);
    data.az = az * (9.81 / 4096.0);
    data.gx = gx * ((PI / 180.0) / 32.8);
    data.gy = gy * ((PI / 180.0) / 32.8);
    data.gz = gz * ((PI / 180.0) / 32.8);
  }

  // Read Mag directly from AK09916 via Bypass
  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x11); // HXL
  I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_MAG, 7, (int)true);

  if (I2C_ICM->available() == 7) {
    int16_t mx = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t my = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t mz = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    I2C_ICM->read(); // ST2

    // Scale: 0.15 uT / LSB
    data.mx = mx * 0.15f;
    data.my = my * 0.15f;
    data.mz = mz * 0.15f;
  }
}

void scanI2C(TwoWire *wire, const char *busName) {
  uint8_t error, address;
  int nDevices = 0;
  Serial.printf("Scanning %s...\n", busName);
  for (address = 1; address < 127; address++) {
    wire->beginTransmission(address);
    error = wire->endTransmission();
    if (error == 0) {
      Serial.printf("I2C device found at address 0x%02X\n", address);
      nDevices++;
    } else if (error == 4) {
      Serial.printf("Unknown error at address 0x%02X\n", address);
    }
  }
  if (nDevices == 0) {
    Serial.println("No I2C devices found.\n");
  } else {
    Serial.println("done\n");
  }
}

void updateOLED(String letter, float accuracy) {
  if (!display)
    return;
  display->clearDisplay();
  display->setTextColor(WHITE);
  display->setTextSize(4);
  display->setCursor(22, 30);
  display->println(letter);

  display->setTextSize(1);
  display->setCursor(5, 90);
  display->print("Acc: ");
  display->print(accuracy, 1);
  display->println("%");
  display->display();
}

void setup() {
  Serial.begin(115200);

  // Allocate TwoWire objects dynamically to prevent early boot crashes
  I2C_MPU = new TwoWire(0);
  I2C_ICM = new TwoWire(1);

  // Initialize Two I2C Buses
  I2C_MPU->begin(I2C0_SDA, I2C0_SCL, 400000); // 21, 22
  I2C_ICM->begin(I2C1_SDA, I2C1_SCL, 400000); // 32, 33

  // Initialize OLED on I2C_MPU (Pins 21/22)
  display =
      new Adafruit_SSD1306(SCREEN_WIDTH, SCREEN_HEIGHT, I2C_MPU, OLED_RESET);
  if (!display->begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED Init Failed"));
  } else {
    display->setRotation(1);
    display->clearDisplay();
    display->setTextSize(2);
    display->setTextColor(WHITE);
    display->setCursor(0, 50);
    display->println("AirWriting");
    display->display();
  }

  Serial.println("=====================================");
  Serial.println("I2C Scanner - Checking Connections...");
  scanI2C(I2C_MPU, "Bus I2C_MPU (Pins 21/22)");
  scanI2C(I2C_ICM, "Bus I2C_ICM (Pins 32/33)");
  Serial.println("=====================================");

  pinMode(PEN_BTN_PIN, INPUT_PULLUP);
  pinMode(VIB_MOTOR_PIN, OUTPUT);
  digitalWrite(VIB_MOTOR_PIN, LOW);

  packet.header = 0xAA;
  packet.footer = 0x55;

  Serial.println("WiFi Connecting...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected.");
  udp.begin(localPort); // Start listening for incoming PC messages

  // Setup Sensors
  setupMPU6050(ADDR_S1_MPU); // S1
  setupMPU6050(ADDR_S2_MPU); // S2
  setupICM20948();           // S3
}

void loop() {
  packet.timestamp = millis();

  readMPU6050(ADDR_S1_MPU, packet.s1); // Forearm
  readMPU6050(ADDR_S2_MPU, packet.s2); // Hand
  readICM20948(packet.s3);             // Finger (ICM20948)

  // -- Button Debounce Logic --
  uint8_t currentRawBtnState = digitalRead(PEN_BTN_PIN);

  if (currentRawBtnState != lastUnstableBtnState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (currentRawBtnState != stableBtnState) {
      stableBtnState = currentRawBtnState;
    }
  }
  lastUnstableBtnState = currentRawBtnState;

  // Set packet based on STABLE state
  packet.button = (stableBtnState == LOW) ? 1 : 0;

  // Haptic Feedback Logic (using STABLE state)
  if (stableBtnState == LOW && lastBtnState == HIGH) {
    // Pen Down Event -> Trigger 40ms Vibration
    digitalWrite(VIB_MOTOR_PIN, HIGH);
    vibTimer = millis();
    isVibActive = true;
  }
  lastBtnState = stableBtnState;

  if (isVibActive && (millis() - vibTimer >= 40)) {
    digitalWrite(VIB_MOTOR_PIN, LOW);
    isVibActive = false;
  }

  uint8_t *ptr = (uint8_t *)&packet;
  uint8_t cksum = 0;
  for (int i = 1; i < 90; i++) { // XOR from Timestamp(1) to Button(89)
    cksum ^= ptr[i];
  }
  packet.checksum = cksum;

  udp.beginPacket(targetIP, targetPort);
  udp.write(ptr, sizeof(AirWritingPacketV3));
  udp.endPacket();

  // Check for incoming UDP message from PC
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char incomingMsg[32];
    int len = udp.read(incomingMsg, 31);
    if (len > 0) {
      incomingMsg[len] = 0;
      String msg = String(incomingMsg);
      int commaIndex = msg.indexOf(',');
      if (commaIndex > 0) {
        String letter = msg.substring(0, commaIndex);
        float accuracy = msg.substring(commaIndex + 1).toFloat();
        updateOLED(letter, accuracy);
      }
    }
  }

  delay(10);
}
