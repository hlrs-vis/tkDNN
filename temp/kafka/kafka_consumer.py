from confluent_kafka import Consumer, KafkaException
import json

class KafkaConsumerWrapper:
    def __init__(self, broker, group_id, topic):
        self.conf = {
            'bootstrap.servers': broker,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.conf)
        self.consumer.subscribe([topic])

    def update(self):
        
        msg = self.consumer.poll(1.0)

        if msg is None:
            return 0

        unprocessed_message = json.loads(msg.value().decode('utf-8'))
        processed_message = self.process_message(unprocessed_message)
        return processed_message

    def process_message(self, json_data):

        print("Received JSON data:", json_data)

class KafkaDetectionConsumer(KafkaConsumerWrapper):
    def process_message(self, json_data):
        # The detections are not yet in a format supported by deepsort
        detections = []

        for value in json_data["detections"]:
            value["frame_id"] = json_data["frame_id"]
            value["timestamp"] = json_data["timestamp"]
            value["cam_id"] = json_data["cam_id"]
            detections.append(value)
        return detections
    
class KafkaCalibrationConsumer(KafkaConsumerWrapper):
    def process_message(self, json_data):
        # Gets the camera calibration data 
        detections = []
        for value in json_data["detections"]:
            value["frame_id"] = json_data["frame_id"]
            value["timestamp"] = json_data["timestamp"]
            value["cam_id"] = json_data["cam_id"]
            detections.append(value)
        return detections