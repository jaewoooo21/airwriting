# AirWriting_ESP32.ino

```cpp
/**
 * AirWriting ESP32 Firmware (Raw Data Transmitter)
 * - MPU-6050 (Address: 0x68)
 * - ICM-20948 (Address: 0x69)
 * - Pen Button: GPIO 15
 * - Sends 80-byte packets matching AirWritingIMUController v2.2 format
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

const char* ssid = "재우의 S25";
const char* password = "asdf750505*";
const char* hostIP = "192.168.x.x"; // PC IP required
const int udpPort = 12345;

WiFiUDP udp;

struct PktFormat {
    uint8_t header = 0xAA;
    uint32_t ts;
    float s1[6]; // acc X,Y,Z, gyro X,Y,Z
    float s2[6];
    float s3[6];
    uint8_t btn;
    uint8_t cksum;
    uint8_t footer = 0x55;
};

void setup() {
    Serial.begin(115200);
    Wire.begin();
    pinMode(15, INPUT_PULLUP);

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
    }
    Serial.println("\nWiFi Connected.");
}

void loop() {
    PktFormat pkt;
    pkt.ts = millis();
    // TODO: I2C Read Logic
    pkt.btn = !digitalRead(15);
    // TODO: calculate cksum and udp.write
    delay(10);
}
```
