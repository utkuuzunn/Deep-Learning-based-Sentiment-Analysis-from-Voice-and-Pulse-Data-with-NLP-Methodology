// Arduino kodu
// nabiz_sensor_kodu.ino
// Amaç: AD8232 sensörüyle nabız ölçmek ve OLED ekranda görüntülemek

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDRESS  0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// AD8232 çıkışı Arduino A0 pinine bağlanır
const int AD8232_Pin = A0;

// Nabız hesaplaması için değişkenler
const int numReadings = 10;
int readings[numReadings] = {0};
int readIndex = 0;
int total = 0;
int average = 0;

unsigned long measureStartTime = 0;
bool measuring = false;

void setup() {
  Serial.begin(9600);

  // OLED başlatılıyor
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println(F("OLED ekran bulunamadı!"));
    while (true);
  }

  display.display();
  delay(2000);
  display.clearDisplay();

  // Okumalar sıfırlanıyor
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;
  }
}

void loop() {
  int sensorValue = analogRead(AD8232_Pin);

  // Sensör bağlı değilse tüm verileri sıfırla
  if (sensorValue < 512) {
    measuring = false;
    measureStartTime = 0;
    total = 0;
    average = 0;
    for (int i = 0; i < numReadings; i++) readings[i] = 0;
  } else {
    // Sensör ilk kez bağlandıysa zamanı başlat
    if (!measuring) {
      measuring = true;
      measureStartTime = millis();
    }
  }

  if (measuring) {
    unsigned long currentMillis = millis();

    // İlk 10 saniyede "ölçüm yapılıyor" mesajı göster
    if (currentMillis - measureStartTime < 10000) {
      display.clearDisplay();
      display.setTextSize(2);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 10);
      display.print("Olcum");
      display.setCursor(0, 40);
      display.print("Yapiliyor");
      display.display();
    } else {
      // Yeni nabız verisini hesapla
      total -= readings[readIndex];
      readings[readIndex] = map(sensorValue, 0, 1023, 60, 100);  // varsayılan aralık
      total += readings[readIndex];
      readIndex = (readIndex + 1) % numReadings;
      average = total / numReadings;

      // OLED ekrana yaz
      display.clearDisplay();
      display.setTextSize(2);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 10);
      display.print("Nabiz:");
      display.setCursor(0, 40);
      display.print(average);
      display.display();
    }
  } else {
    // Sensör bağlı değilken gösterilen mesaj
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 10);
    display.print("Olcum");
    display.setCursor(0, 40);
    display.print("Yapilmiyor");
    display.display();
  }

  // Seri monitöre yaz
  Serial.print("Nabiz: ");
  Serial.println(average);

  delay(100);  // okuma frekansı
}
