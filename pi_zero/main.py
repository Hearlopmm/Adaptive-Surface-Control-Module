#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt

def angle_to_duty(angle):
    angle = max(0, min(180, angle))  # 限制范围
    pulse_ms = 0.5 + (angle / 180.0) * 2.0  # 0.5 ~ 2.5 ms
    duty = pulse_ms / 20.0 * 100  # 20ms 周期 => 50Hz
    return duty

def set_angle(pwm, angle):
    duty = angle_to_duty(angle)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.4)
    pwm.ChangeDutyCycle(0)  # 停掉占空比，减抖动

GPIO.setmode(GPIO.BCM)

PIN12 = 12
PIN13 = 13
PIN19 = 19

for pin in (PIN12, PIN13, PIN19):
    GPIO.setup(pin, GPIO.OUT)

freq = 50
pwm12 = GPIO.PWM(PIN12, freq)
pwm13 = GPIO.PWM(PIN13, freq)
pwm19 = GPIO.PWM(PIN19, freq)

pwm12.start(0)
pwm13.start(0)
pwm19.start(0)

# 当前角度状态（初始化为 45°）
angle12 = 45
angle13 = 45
angle19 = 45

set_angle(pwm12, angle12)
set_angle(pwm13, angle13)
set_angle(pwm19, angle19)

print("Servo program (Zero) started.")

# ================== MQTT ==================
MQTT_BROKER_HOST = "192.168.137.239"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC       = "demo/number"
MQTT_CLIENT_ID   = "pi_zero_servo"

# ================== MQTT 回调 ==================
def on_connect(client, userdata, flags, rc):
    print("Connected to broker, rc =", rc)
    client.subscribe(MQTT_TOPIC)
    print("Subscribed to:", MQTT_TOPIC)

def on_message(client, userdata, msg):
    global angle12, angle13, angle19

    payload = msg.payload.decode("utf-8").strip()
    if not payload:
        return

    print(f"\n[MQTT] topic={msg.topic}, payload='{payload}'")

    if payload == "0":
        if angle12 != 45:
            angle12 = 45
            set_angle(pwm12, angle12)
            print("GPIO12 -> 45° (reset)")
        if angle13 != 45:
            angle13 = 45
            set_angle(pwm13, angle13)
            print("GPIO13 -> 45° (reset)")
        if angle19 != 45:
            angle19 = 45
            set_angle(pwm19, angle19)
            print("GPIO19 -> 45° (reset)")
        return

    for ch in payload:
        if ch == '1':
            # 12: 45 <-> 130
            if angle12 == 45:
                angle12 = 130
            else:
                angle12 = 45
            set_angle(pwm12, angle12)
            print(f"GPIO12 -> {angle12}°")

        elif ch == '2':
            # 13: 45 <-> 130
            if angle13 == 45:
                angle13 = 130
            else:
                angle13 = 45
            set_angle(pwm13, angle13)
            print(f"GPIO13 -> {angle13}°")

        elif ch == '3':
            if angle19 == 45:
                angle19 = 130
            else:
                angle19 = 45
            set_angle(pwm19, angle19)
            print(f"GPIO19 -> {angle19}°")

        else:
            print(f"Unknown command char: '{ch}', ignored.")

# ================== 主程序 ==================
client = mqtt.Client(client_id=MQTT_CLIENT_ID)  # 旧版 paho-mqtt 写法
client.on_connect = on_connect
client.on_message = on_message

while True:
    try:
        print(f"Trying to connect to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} ...")
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        print("MQTT connected.")
        break
    except Exception as e:
        print("Connect failed:", e)
        print("Retry in 5 seconds...")
        time.sleep(5)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("KeyboardInterrupt, exiting...")
finally:
    pwm12.stop()
    pwm13.stop()
    pwm19.stop()
    GPIO.cleanup()
    print("Clean exit.")
