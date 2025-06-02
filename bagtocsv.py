import os
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
import rosbag2_py


def read_messages(input_bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp


def write_csv(bag_path):
    messages = {}
    for topic, msg, timestamp in read_messages(bag_path):
        if topic not in messages:
            messages[topic] = []
        messages[topic].append({'time': timestamp / 1e9, 'data': msg})  # Zeit in Sekunden

    for topic, msgs in messages.items():
        rows = []
        for m in msgs:
            msg_dict = message_to_ordereddict(m['data'])
            flat_dict = flatten_dict(msg_dict)
            flat_dict['time'] = m['time']
            rows.append(flat_dict)
        df = pd.DataFrame(rows)
        topic_name = topic.strip('/').replace('/', '_')
        df.to_csv(f"{topic_name}.csv", index=False)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    bag_path = "rosbag2_2025_06_02-15_26_49_0.mcap"
    write_csv(bag_path)

