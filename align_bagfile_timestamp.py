import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from rosbag2_py import SequentialReader, SequentialWriter
from rosbag2_py._storage import StorageOptions, ConverterOptions, TopicMetadata
import importlib
import sys

def align_exact_timestamps(input_bag_path, output_bag_path, topic1, topic2):
    rclpy.init()
    node = rclpy.create_node('bag_modifier')

    reader = SequentialReader()
    storage_options = StorageOptions(uri=input_bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    writer = SequentialWriter()
    storage_options_out = StorageOptions(uri=output_bag_path, storage_id='sqlite3')
    writer.open(storage_options_out, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_info_map = {
        topic.name: TopicMetadata(name=topic.name, type=topic.type, serialization_format='cdr')
        for topic in topic_types
    }

    for topic_metadata in topic_info_map.values():
        writer.create_topic(topic_metadata)

    # Read all messages and group them by topics
    messages = {topic1: [], topic2: []}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic in messages:
            msg_type = [x.type for x in topic_types if x.name == topic][0]
            module_name, class_name = msg_type.rsplit('/', 1)
            msg_module = importlib.import_module(module_name.replace('/', '.'))
            msg_class = getattr(msg_module, class_name)
            msg = deserialize_message(data, msg_class)
            messages[topic].append((timestamp, msg))

    # Find minimum message count for alignment
    min_message_count = min(len(messages[topic1]), len(messages[topic2]))
    aligned_messages = []

    for i in range(min_message_count):
        ts1, msg1 = messages[topic1][i]
        ts2, msg2 = messages[topic2][i]

        # Use the average of the two timestamps as the new common timestamp
        common_timestamp = (ts1 + ts2) // 2

        # Update timestamps in the messages
        if hasattr(msg1, 'header') and hasattr(msg1.header, 'stamp'):
            msg1.header.stamp.sec = common_timestamp // 10**9
            msg1.header.stamp.nanosec = common_timestamp % 10**9

        if hasattr(msg2, 'header') and hasattr(msg2.header, 'stamp'):
            msg2.header.stamp.sec = common_timestamp // 10**9
            msg2.header.stamp.nanosec = common_timestamp % 10**9

        aligned_messages.append((topic1, common_timestamp, msg1))
        aligned_messages.append((topic2, common_timestamp, msg2))

    # Write aligned messages to the new bag
    for topic, timestamp, msg in aligned_messages:
        writer.write(topic, serialize_message(msg), timestamp)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python align_exact_camera_topics.py <input_bag> <output_bag>")
        sys.exit(1)

    input_bag_path = sys.argv[1]
    output_bag_path = sys.argv[2]
    topic1 = '/camera1/image_raw'
    topic2 = '/camera2/image_raw'

    align_exact_timestamps(input_bag_path, output_bag_path, topic1, topic2)
