from confluent_kafka import Consumer
import json

c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest'
})

c.subscribe(['timed-images'])

while True:
    msg = c.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue
    
    try:
        json_message = json.loads(msg.value().decode('utf-8'))
        string1 = json_message["str1"]
        string2 = json_message["str2"]
        print(f"Received strings: {string1}, {string2}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

c.close()
