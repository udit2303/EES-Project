#include<Servo.h>
Servo servo;
int x=0; 
void setup() {
  Serial.begin(9600);
  servo.attach(9);
  Serial.println("STarted");
  servo.write(105);
}
 
void loop() {
  int y= Serial.readString().toInt();
  Serial.println(1);
  if(y==1){ //Detected as rotten
    servo.write(150);
    delay(1000);
    servo.write(105);
    y=0;
  }
  else if(y==2){
    servo.write(60);
    delay(1000);
    servo.write(105);
   y=0;
  }
}