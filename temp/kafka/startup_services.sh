#!/bin/bash


gnome-terminal --title="Zookeeper Server" --working-directory="/home/visadmin/Desktop/Kafka" -- bash -c 'bin/zookeeper-server-start.sh ./config/zookeeper.properties'
sleep 5

gnome-terminal --title="Kafka Server" --working-directory="/home/visadmin/Desktop/Kafka" -- bash -c 'bin/kafka-server-start.sh ./config/server.properties'

sleep 5  
gnome-terminal --title="Terminal 3" --working-directory="/home/visadmin/Desktop/Kafka" -- bash -c 'bin/kafka-topics.sh --create --topic timed-images --bootstrap-server localhost:9092; read -p "Press Enter to close..."'
