// AirWriting_ESP32_v3.ino — Hybrid-AirScribe Dual-Node
// WiFi & IMU UDP streaming for ESP32-S3
// - S1 (WRIST): MPU6050 on I2C bus 0 (0x68)
// - S2 (HAND):  [DISABLED] MPU6050 on I2C bus 0 (0x69) — 주석처리됨, 복구 가능
// - S3 (FINGER): ICM20948 + AK09916 on I2C bus 1 (0x68)
// - WiFiManager captive portal + mDNS + UDP broadcast discovery

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <WiFiUdp.h>
#include <Wire.h>

const char *ssid = "재우의S25";   
const char *pass = "kpu123456!";

// UDP configuration
const char *fallbackTargetIP = "172.30.1.72";  // 데스크탑 이더넷 IP
const int targetPort = 12345;
const int localPort = 5555;
const int discoveryPort = 12344;
const char *discoveryRequest = "AIRWRITING_DISCOVER_V1";

WiFiUDP udp;
IPAddress targetIP;
bool serverDiscovered = false;
unsigned long lastDiscoveryAttemptMs = 0;
String serialCommandBuffer;

struct SensorData6;
struct SensorData9;
struct AirWritingPacketV4;

void setOLEDLinkState(String bannerRaw, String hintRaw, bool redraw = true);
void renderOLEDStatus(String stateRaw, String modeRaw, String headlineRaw, float accuracy);
void readMPU6050(uint8_t addr, SensorData6 &data);
void readICM20948(SensorData9 &data);

void handleSerialConfigInput() {}

IPAddress getBroadcastIP() {
  IPAddress broadcast(255, 255, 255, 255);
  IPAddress localIp = WiFi.localIP();
  IPAddress subnet = WiFi.subnetMask();
  for (int i = 0; i < 4; i++) {
    broadcast[i] = localIp[i] | (~subnet[i]);
  }
  return broadcast;
}

bool parseDiscoveryResponse(const String &msg) {
  const String prefix = "AIRWRITING_SERVER_V1 ";
  if (!msg.startsWith(prefix)) {
    return false;
  }

  int secondSpace = msg.indexOf(' ', prefix.length());
  if (secondSpace < 0) {
    return false;
  }

  String ipText = msg.substring(prefix.length(), secondSpace);
  String portText = msg.substring(secondSpace + 1);
  IPAddress discoveredIp;
  if (!discoveredIp.fromString(ipText)) {
    return false;
  }

  int announcedPort = portText.toInt();
  if (announcedPort <= 0) {
    return false;
  }

  targetIP = discoveredIp;
  serverDiscovered = true;
  setOLEDLinkState("JETSON WAIT", "AWAIT RUNTIME");
  Serial.print("Discovered server: ");
  Serial.print(targetIP);
  Serial.print(":");
  Serial.println(announcedPort);
  return true;
}

bool discoverServerViaMDNS() {
  setOLEDLinkState("MDNS FIND", "AUTO SEARCH");
  Serial.println("[Discovery] Trying mDNS service query...");

  int n = MDNS.queryService("airwriting", "udp");
  if (n > 0) {
    targetIP = MDNS.address(0);
    int port = MDNS.port(0);
    serverDiscovered = true;
    setOLEDLinkState("MDNS OK", "AUTO FOUND");
    Serial.print("[Discovery] Found server via mDNS: ");
    Serial.print(targetIP);
    Serial.print(":");
    Serial.println(port);
    return true;
  }

  Serial.println("[Discovery] mDNS: no service found");
  return false;
}

bool discoverServerViaBroadcast(unsigned long timeoutMs = 3000) {
  IPAddress broadcastIp = getBroadcastIP();
  unsigned long startedAt = millis();
  setOLEDLinkState("BCAST FIND", "SEND BCAST");
  Serial.println("[Discovery] Trying UDP broadcast...");

  while (millis() - startedAt < timeoutMs) {
    udp.beginPacket(broadcastIp, discoveryPort);
    udp.write((const uint8_t *)discoveryRequest, strlen(discoveryRequest));
    udp.endPacket();

    unsigned long waitUntil = millis() + 400;
    while (millis() < waitUntil) {
      int packetSize = udp.parsePacket();
      if (!packetSize) {
        delay(20);
        continue;
      }

      char incomingMsg[96];
      int len = udp.read(incomingMsg, sizeof(incomingMsg) - 1);
      if (len <= 0) {
        continue;
      }

      incomingMsg[len] = 0;
      if (parseDiscoveryResponse(String(incomingMsg))) {
        Serial.println("[Discovery] Found server via broadcast");
        return true;
      }
    }
  }

  Serial.println("[Discovery] broadcast: no response");
  return false;
}

bool discoverServer(unsigned long broadcastTimeoutMs = 3000) {
  if (discoverServerViaMDNS()) {
    return true;
  }

  if (discoverServerViaBroadcast(broadcastTimeoutMs)) {
    return true;
  }

  IPAddress fallbackIp;
  if (fallbackIp.fromString(fallbackTargetIP)) {
    targetIP = fallbackIp;
    setOLEDLinkState("FALLBACK", "USE SAVED IP");
    Serial.print("[Discovery] Using saved fallback IP: ");
    Serial.println(targetIP);
    return false;
  }

  setOLEDLinkState("NO SERVER", "CHECK WIFI");
  Serial.println("[Discovery] All methods failed");
  return false;
}

// OLED configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 *display;

// Hardware pins
const int PEN_BTN_PIN = 15;
const int VIB_MOTOR_PIN = 4;

// Debounce and haptic state
uint8_t lastBtnState = 1;
unsigned long vibTimer = 0;
bool isVibActive = false;

// Debounce variables
uint8_t stableBtnState = 1;
uint8_t lastUnstableBtnState = 1;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 100;
String oledState = "BOOT";
String oledMode = "BOOT";
String oledHeadline = "AIR";
float oledAccuracy = 0.0f;
String oledLinkBanner = "WIFI JOIN";
String oledLinkHint = "CONNECT AP";
unsigned long lastStatusMessageMs = 0;
const unsigned long oledStatusTimeoutMs = 4000;
bool oledTimeoutRendered = false;

// I2C bus pins
const int I2C0_SDA = 21;
const int I2C0_SCL = 22;
const int I2C1_SDA = 32;
const int I2C1_SCL = 33;

TwoWire *I2C_MPU;
TwoWire *I2C_ICM;

// IMU addresses per bus — Dual-Node (WRIST + FINGER)
const int ADDR_S1_MPU = 0x68;   // WRIST (손목 / 전완근) — Bus 0
const int ADDR_S2_MPU = 0x69;   // HAND (손등) — Bus 0
const int ADDR_S3_ICM = 0x68;   // FINGER (손가락) — Bus 1
const int ADDR_MAG = 0x0C;      // AK09916 magnetometer — Bus 1

#pragma pack(push, 1)
struct SensorData6 {
  float ax, ay, az;
  float gx, gy, gz;
};

struct SensorData9 {
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
};

struct AirWritingPacketV4 {
  uint8_t header;              // 0xAA
  uint16_t seq;                // 시퀀스 번호 (패킷 유실 감지용)
  uint32_t timestamp;          // millis()
  SensorData6 s1;              // WRIST (전완근) — 24 bytes
  SensorData6 s2;              // HAND (손등) — 24 bytes
  SensorData9 s3;              // FINGER (손가락) — 36 bytes
  uint8_t button;
  uint8_t checksum;
  uint8_t footer;              // 0x55
};
// 패킷 크기: 1+2+4+24+24+36+1+1+1 = 94 bytes
#pragma pack(pop)

AirWritingPacketV4 packet;
uint16_t packetSeq = 0;

// ── 링 버퍼 (WiFi jitter 흡수) ──
const int RING_BUF_SIZE = 16;
AirWritingPacketV4 ringBuffer[RING_BUF_SIZE];
volatile int rbHead = 0;       // 쓰기 위치 (ISR/loop에서 증가)
volatile int rbTail = 0;       // 읽기 위치 (전송 루프에서 증가)

int rbCount() { return (rbHead - rbTail + RING_BUF_SIZE) % RING_BUF_SIZE; }
bool rbFull()  { return rbCount() == RING_BUF_SIZE - 1; }
bool rbEmpty() { return rbHead == rbTail; }

void rbPush(const AirWritingPacketV4 &item) {
  ringBuffer[rbHead] = item;
  rbHead = (rbHead + 1) % RING_BUF_SIZE;
  if (rbHead == rbTail) { // overflow → 가장 오래된 것 버림
    rbTail = (rbTail + 1) % RING_BUF_SIZE;
  }
}

AirWritingPacketV4* rbPeek() {
  if (rbEmpty()) return nullptr;
  return &ringBuffer[rbTail];
}

void rbPop() {
  if (!rbEmpty()) rbTail = (rbTail + 1) % RING_BUF_SIZE;
}

// ── 100Hz 정밀 타이밍 ──
unsigned long lastSampleMicros = 0;
const unsigned long SAMPLE_INTERVAL_US = 10000; // 100Hz = 10ms

void setICMBank(uint8_t bank) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x7F);
  I2C_ICM->write(bank << 4);
  I2C_ICM->endTransmission(true);
}

void setupMPU6050(uint8_t addr) {
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x6B);
  I2C_MPU->write(0x01);
  I2C_MPU->endTransmission(true);

  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1C);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);

  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1B);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);
}

void setupICM20948() {
  setICMBank(0);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x06); // ICM-20948 PWR_MGMT_1 (Bank 0)
  I2C_ICM->write(0x01); // Auto Clock Select, Wake up
  I2C_ICM->endTransmission(true);
  delay(10);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x03); // USER_CTRL (Bank 0)
  I2C_ICM->write(0x00); // I2C Master 비활성화 (Bypass를 위해)
  I2C_ICM->endTransmission(true);
  delay(10);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x0F); // INT_PIN_CFG (Bank 0)
  I2C_ICM->write(0x02); // BYPASS_EN 켜기 (Mag 직접 접근용)
  I2C_ICM->endTransmission(true);
  delay(10);

  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x31); // CNTL2 (AK09916)
  I2C_ICM->write(0x08); // 100Hz 연속 측정
  I2C_ICM->endTransmission(true);
  delay(10);

  setICMBank(2);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x14); // ICM-20948 ACCEL_CONFIG (Bank 2)
  I2C_ICM->write(0x04); // ±8g (비트 [2:1] = 10)
  I2C_ICM->endTransmission(true);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x01); // ICM-20948 GYRO_CONFIG_1 (Bank 2)
  I2C_ICM->write(0x04); // ±1000dps (비트 [2:1] = 10)
  I2C_ICM->endTransmission(true);

  setICMBank(0);
}

void readMPU6050(uint8_t addr, SensorData6 &data) {
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x3B);
  I2C_MPU->endTransmission(false);
  I2C_MPU->requestFrom((int)addr, 14, (int)true);

  if (I2C_MPU->available() == 14) {
    int16_t ax = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t ay = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t az = (I2C_MPU->read() << 8 | I2C_MPU->read());
    I2C_MPU->read();
    I2C_MPU->read();
    int16_t gx = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gy = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gz = (I2C_MPU->read() << 8 | I2C_MPU->read());

    data.ax = ax * (9.81f / 4096.0f);
    data.ay = ay * (9.81f / 4096.0f);
    data.az = az * (9.81f / 4096.0f);
    data.gx = gx * ((PI / 180.0f) / 32.8f);
    data.gy = gy * ((PI / 180.0f) / 32.8f);
    data.gz = gz * ((PI / 180.0f) / 32.8f);
  } else {
    data.ax = data.ay = data.az = 0;
    data.gx = data.gy = data.gz = 0;
  }
}

void readICM20948(SensorData9 &data) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x2D);
  I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_S3_ICM, 14, (int)true);

  if (I2C_ICM->available() == 14) {
    int16_t ax = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t ay = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t az = (I2C_ICM->read() << 8 | I2C_ICM->read());
    I2C_ICM->read();
    I2C_ICM->read();
    int16_t gx = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gy = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gz = (I2C_ICM->read() << 8 | I2C_ICM->read());

    data.ax = ax * (9.81f / 4096.0f);
    data.ay = ay * (9.81f / 4096.0f);
    data.az = az * (9.81f / 4096.0f);
    data.gx = gx * ((PI / 180.0f) / 32.8f);
    data.gy = gy * ((PI / 180.0f) / 32.8f);
    data.gz = gz * ((PI / 180.0f) / 32.8f);
  } else {
    data.ax = data.ay = data.az = 0;
    data.gx = data.gy = data.gz = 0;
  }

  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x11);
  I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_MAG, 7, (int)true);

  if (I2C_ICM->available() == 7) {
    int16_t mx = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t my = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t mz = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    I2C_ICM->read();

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

String sanitizeOLEDField(String value, String fallback, size_t maxLen) {
  value.trim();
  value.toUpperCase();
  value.replace("|", "/");
  value.replace(",", " ");
  if (value.length() == 0) {
    value = fallback;
  }
  if (value.length() > maxLen) {
    value = value.substring(0, maxLen);
  }
  return value;
}

String oledDetailForState(const String &state) {
  if (state == "CAL") return "Calibrating";
  if (state == "READY") return "Connected";
  if (state == "WRITE") return "Writing";
  if (state == "OK") return "Recognized";
  if (state == "PEND") return "Confirm on phone";
  if (state == "REJ") return "Rejected";
  if (state == "FAIL") return "Dispatch failed";
  if (state == "CXL") return "Cancelled";
  if (state == "EXP") return "Expired";
  return "Runtime update";
}

void setOLEDLinkState(String bannerRaw, String hintRaw, bool redraw) {
  oledLinkBanner = sanitizeOLEDField(bannerRaw, "JETSON FIND", 12);
  oledLinkHint = sanitizeOLEDField(hintRaw, "CHECK WIFI", 18);
  if (redraw && display) {
    renderOLEDStatus(oledState, oledMode, oledHeadline, oledAccuracy);
  }
}

void renderOLEDStatus(String stateRaw, String modeRaw, String headlineRaw, float accuracy) {
  if (!display) {
    return;
  }

  oledState = sanitizeOLEDField(stateRaw, "READY", 8);
  oledMode = sanitizeOLEDField(modeRaw, "NONE", 12);
  oledHeadline = sanitizeOLEDField(headlineRaw, oledState, 12);
  oledAccuracy = accuracy < 0.0f ? 0.0f : accuracy;

  const bool jetsonOnline =
      lastStatusMessageMs > 0 && (millis() - lastStatusMessageMs) < oledStatusTimeoutMs;
  String connectionText = jetsonOnline ? "JETSON ON" : oledLinkBanner;

  const String modeText = "MODE " + oledMode;
  const String detailText = jetsonOnline ? oledDetailForState(oledState) : oledLinkHint;

  uint8_t mainSize = 4;
  if (oledHeadline.length() >= 5) {
    mainSize = 2;
  } else if (oledHeadline.length() >= 3) {
    mainSize = 3;
  }

  const int screenWidth = display->width();
  const int screenHeight = display->height();

  display->clearDisplay();
  display->setTextColor(WHITE);
  display->setTextSize(1);
  display->setCursor(0, 0);
  display->println(connectionText);
  display->setCursor(0, 10);
  display->println(modeText);
  display->drawLine(0, 20, screenWidth - 1, 20, WHITE);

  display->setTextSize(mainSize);
  int16_t x1, y1;
  uint16_t textW, textH;
  display->getTextBounds(oledHeadline, 0, 0, &x1, &y1, &textW, &textH);
  int mainX = (screenWidth - static_cast<int>(textW)) / 2;
  if (mainX < 0) {
    mainX = 0;
  }
  const int mainY = screenHeight > 96 ? 34 : 24;
  display->setCursor(mainX, mainY);
  display->println(oledHeadline);

  display->setTextSize(1);
  display->setCursor(0, screenHeight - 24);
  display->println(detailText);
  display->setCursor(0, screenHeight - 12);
  if (oledAccuracy > 0.05f) {
    display->print("Acc ");
    display->print(oledAccuracy, 1);
    display->print("%");
  } else {
    display->print("Acc -");
  }
  display->display();
}

bool parseOLEDStatusMessage(const String &msg) {
  const String prefix = "OLED|";
  if (!msg.startsWith(prefix)) {
    return false;
  }

  const int first = msg.indexOf('|', prefix.length());
  const int second = first >= 0 ? msg.indexOf('|', first + 1) : -1;
  const int third = second >= 0 ? msg.indexOf('|', second + 1) : -1;
  if (first < 0 || second < 0 || third < 0) {
    return false;
  }

  const String state = msg.substring(prefix.length(), first);
  const String mode = msg.substring(first + 1, second);
  const String headline = msg.substring(second + 1, third);
  const float accuracy = msg.substring(third + 1).toFloat();
  lastStatusMessageMs = millis();
  oledTimeoutRendered = false;
  renderOLEDStatus(state, mode, headline, accuracy);
  return true;
}

void updateOLED(String letter, float accuracy) {
  lastStatusMessageMs = millis();
  oledTimeoutRendered = false;
  renderOLEDStatus("OK", oledMode, letter, accuracy);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Starting AirWriting v3 (Hardcoded)...");

  I2C_MPU = new TwoWire(0);
  I2C_ICM = new TwoWire(1);

  I2C_MPU->begin(I2C0_SDA, I2C0_SCL, 400000);
  I2C_ICM->begin(I2C1_SDA, I2C1_SCL, 400000);

  display = new Adafruit_SSD1306(SCREEN_WIDTH, SCREEN_HEIGHT, I2C_MPU, OLED_RESET);
  if (!display->begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED Init Failed"));
  } else {
    display->setRotation(1);
    setOLEDLinkState("WIFI JOIN", "CONNECT AP", false);
    renderOLEDStatus("BOOT", "BOOT", "READY", 0.0f);
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
  Serial.println("USB Serial Mode Active!");
  setOLEDLinkState("USB SERIAL", "CONNECTED", false);
  
  setupMPU6050(ADDR_S1_MPU);      // WRIST 초기화
  setupMPU6050(ADDR_S2_MPU);      // HAND 초기화
  setupICM20948();                 // FINGER 초기화
}

void loop() {
  handleSerialConfigInput();

  // ── 100Hz 정밀 센서 샘플링 ──
  unsigned long nowMicros = micros();
  if (nowMicros - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;
    // 큰 지연이 있었으면 현재 시간으로 리셋 (스킵 방지)
    if (nowMicros - lastSampleMicros > SAMPLE_INTERVAL_US * 3) {
      lastSampleMicros = nowMicros;
    }

    packet.header = 0xAA;
    packet.seq = packetSeq++;
    packet.timestamp = millis();

    readMPU6050(ADDR_S1_MPU, packet.s1);      // WRIST 읽기 (전완근)
    readMPU6050(ADDR_S2_MPU, packet.s2);      // HAND 읽기 (손등)
    readICM20948(packet.s3);                   // FINGER 읽기 (손가락)

    // 버튼 디바운스
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
    packet.button = (stableBtnState == LOW) ? 1 : 0;

    // 진동 피드백
    if (stableBtnState == LOW && lastBtnState == HIGH) {
      digitalWrite(VIB_MOTOR_PIN, HIGH);
      vibTimer = millis();
      isVibActive = true;
    }
    lastBtnState = stableBtnState;

    // 체크섬
    uint8_t *ptr = (uint8_t *)&packet;
    uint8_t cksum = 0;
    for (int i = 1; i < (int)sizeof(AirWritingPacketV4) - 2; i++) {
      cksum ^= ptr[i];
    }
    packet.checksum = cksum;
    packet.footer = 0x55;

    // 링 버퍼에 저장
    rbPush(packet);
  }

  // 진동 타이머
  if (isVibActive && (millis() - vibTimer >= 40)) {
    digitalWrite(VIB_MOTOR_PIN, LOW);
    isVibActive = false;
  }

  // ── 링 버퍼에서 전송 (USB Serial) ──
  while (!rbEmpty()) {
    AirWritingPacketV4 *pkt = rbPeek();
    // 0xAA (header), 94 bytes struct, 0x55 (footer)
    Serial.write((uint8_t *)pkt, sizeof(AirWritingPacketV4));
    rbPop();
  }
  if (lastStatusMessageMs > 0 && (millis() - lastStatusMessageMs) >= oledStatusTimeoutMs) {
    if (!oledTimeoutRendered) {
      oledTimeoutRendered = true;
      renderOLEDStatus(oledState, oledMode, oledHeadline, oledAccuracy);
    }
  } else {
    oledTimeoutRendered = false;
  }

  delay(10);
}
