// AirWriting_ESP32.ino
// WiFi & IMU UDP Streaming
// - SSID: 재우의 S25
// - Password: asdf750505*

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// WiFi credentials
const char* ssid = "재우의 S25";
const char* pass = "asdf750505*";

// UDP Configuration
const char* targetIP = "192.168.0.0"; // REPLACE WITH PC IP
const int targetPort = 12345;
WiFiUDP udp;

// Hardware Pins
const int PEN_BTN_PIN = 15;

// IMU I2C Addresses
const int ADDR_MPU  = 0x68; // S2 (Forearm) & S3 (Hand) - Assuming one is AD0 high?
const int ADDR_ICM  = 0x69; // S1 (Upper Arm)

#pragma pack(push, 1)
struct SensorData {
  float ax, ay, az;
  float gx, gy, gz;
};

struct AirWritingPacketV2 {
  uint8_t header;         // 1B (0xAA)
  uint32_t timestamp;     // 4B
  SensorData s1;          // 24B
  SensorData s2;          // 24B
  SensorData s3;          // 24B
  uint8_t button;         // 1B
  uint8_t checksum;       // 1B
  uint8_t footer;         // 1B (0x55)
};
#pragma pack(pop)

AirWritingPacketV2 packet;

void setupIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0);    // Wake up
  Wire.endTransmission(true);

  // Set ranges (Acc: 8g, Gyro: 1000 dps)
  Wire.beginTransmission(addr);
  Wire.write(0x1C); // ACCEL_CONFIG
  Wire.write(0x10); // 8g = 4096 LSB/g
  Wire.endTransmission(true);

  Wire.beginTransmission(addr);
  Wire.write(0x1B); // GYRO_CONFIG
  Wire.write(0x10); // 1000 dps = 32.8 LSB/deg/s
  Wire.endTransmission(true);
}

void readIMU(uint8_t addr, SensorData& data) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B); // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom((int)addr, 14, (int)true);

  if (Wire.available() == 14) {
    int16_t ax = (Wire.read() << 8 | Wire.read());
    int16_t ay = (Wire.read() << 8 | Wire.read());
    int16_t az = (Wire.read() << 8 | Wire.read());
    Wire.read(); Wire.read(); // Skip Temp
    int16_t gx = (Wire.read() << 8 | Wire.read());
    int16_t gy = (Wire.read() << 8 | Wire.read());
    int16_t gz = (Wire.read() << 8 | Wire.read());

    // Scale to m/s^2 and rad/s based on 8g and 1000dps
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

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C
  
  pinMode(PEN_BTN_PIN, INPUT_PULLUP);

  // Init Packet constants
  packet.header = 0xAA;
  packet.footer = 0x55;

  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  // Setup IMUs (Example: If you have multiplexer or separate pins, configure here)
  setupIMU(ADDR_MPU);
  setupIMU(ADDR_ICM);
}

void loop() {
  packet.timestamp = millis();

  readIMU(ADDR_ICM, packet.s1);
  readIMU(ADDR_MPU, packet.s2);
  readIMU(ADDR_MPU, packet.s3); // NOTE: Requires TCA9548A or AD0 pin toggle in hardware if both are MPU6050

  packet.button = (digitalRead(PEN_BTN_PIN) == LOW) ? 1 : 0;

  // Calculate Checksum (XOR all bytes between Header(0) and Checksum(78))
  uint8_t* ptr = (uint8_t*)&packet;
  uint8_t cksum = 0;
  for (int i = 1; i <= 77; i++) {
    cksum ^= ptr[i];
  }
  packet.checksum = cksum;

  // Send packet
  udp.beginPacket(targetIP, targetPort);
  udp.write(ptr, sizeof(AirWritingPacketV2)); // should be exactly 80 bytes
  udp.endPacket();

  delay(10); // ~100Hz
}
