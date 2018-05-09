#include <Wire.h>
#include <Servo.h>

// I2Cアドレス
#define DRV_ADR1   0x60  // DRV8830のI2Cアドレス(1つ目)
#define DRV_ADR2   0x68  // DRV8830のI2Cアドレス(2つ目)
#define CTR_ADR   0x00  // CONTROLレジスタのサブアドレス
#define FLT_ADR   0x01  // FAULTレジスタのアドレス

// ブリッジ制御
#define M_STANBY  B00   // スタンバイ   
#define M_REVERSE B01   // 逆転
#define M_NORMAL  B10   // 正転
#define M_BRAKE   B11   // ブレーキ

// 電圧定義
#define MAX_VSET 0x29   // 3.29V
#define MIN_VSET 0x09   // 0.72V

int angle = 0, mot_sta = 0, power = 0, i = 0;
char buff[30];

Servo servo0;

// 制御コマンド送信
void write_vset(byte drv_addr, byte vs, byte ctr) {
  Wire.beginTransmission(drv_addr);
  Wire.write(CTR_ADR);
  Wire.write((vs << 2) | ctr );
  //return Wire.endTransmission();
  Wire.endTransmission(true);
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  servo0.attach(9);
}

void loop() {
  if (Serial.available() > 0) {
    buff[i] = Serial.read();
    if (buff[i] == 'e') {
      buff[i] == '\0';
      angle = atoi(strtok(buff, ","));
      mot_sta = atoi(strtok(NULL, ","));
      power = atoi(strtok(NULL, ","));
      write_vset(DRV_ADR2, power, mot_sta);
      servo0.write(angle+80);
      i = 0;
      
    } else {
      i += 1;
    }
  }
}
