#define LED_PIN 4  // A2 = GPIO4
int brightness = 255; //17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255

void setup() {
  Serial.begin(115200);
  delay(1000);
  analogWrite(LED_PIN, brightness);
  Serial.print("LED brightness set to: ");
  Serial.println(brightness);
}

void loop() {
  // nothing needed if LED stays constant
}
