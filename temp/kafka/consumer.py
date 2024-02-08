from kafka_consumer import KafkaDetectionConsumer
import json

broker = "localhost:9092"
group_id = "1"
topic = "timed-images"

kafka_consumer = KafkaDetectionConsumer(broker, group_id, topic)


while True:
    frame = kafka_consumer.update()
    if frame != 0:
        print(f"My message = {frame}")