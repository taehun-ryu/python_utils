import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from rosbag2_py import SequentialReader, SequentialWriter
from rosbag2_py._storage import StorageOptions, ConverterOptions, TopicMetadata
import importlib
import sys

def align_to_reference_topic(input_bag_path, output_bag_path, reference_topic, target_topic):
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
    reference_messages = []
    target_messages = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == reference_topic:
            msg_type = [x.type for x in topic_types if x.name == topic][0]
            module_name, class_name = msg_type.rsplit('/', 1)
            msg_module = importlib.import_module(module_name.replace('/', '.'))
            msg_class = getattr(msg_module, class_name)
            msg = deserialize_message(data, msg_class)
            reference_messages.append((timestamp, msg))
        elif topic == target_topic:
            msg_type = [x.type for x in topic_types if x.name == topic][0]
            module_name, class_name = msg_type.rsplit('/', 1)
            msg_module = importlib.import_module(module_name.replace('/', '.'))
            msg_class = getattr(msg_module, class_name)
            msg = deserialize_message(data, msg_class)
            target_messages.append((timestamp, msg))

    # Align target messages to reference messages
    aligned_messages = []
    min_message_count = min(len(reference_messages), len(target_messages))

    for i in range(min_message_count):
        ref_timestamp, ref_msg = reference_messages[i]
        _, target_msg = target_messages[i]

        # Update the target message's timestamp to match the reference
        if hasattr(target_msg, 'header') and hasattr(target_msg.header, 'stamp'):
            target_msg.header.stamp.sec = ref_timestamp // 10**9
            target_msg.header.stamp.nanosec = ref_timestamp % 10**9

        # Add both messages to the aligned list
        aligned_messages.append((reference_topic, ref_timestamp, ref_msg))
        aligned_messages.append((target_topic, ref_timestamp, target_msg))

    # Write aligned messages to the new bag
    for topic, timestamp, msg in aligned_messages:
        writer.write(topic, serialize_message(msg), timestamp)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python align_to_reference_topic.py <input_bag> <output_bag>")
        sys.exit(1)

    input_bag_path = sys.argv[1]
    output_bag_path = sys.argv[2]
    reference_topic = '/camera1/image_raw'
    target_topic = '/camera2/image_raw'

    align_to_reference_topic(input_bag_path, output_bag_path, reference_topic, target_topic)
