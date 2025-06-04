# ROS2-bag-to-csv-mcap-to-csv

0. istall dependencies
  $ sudo apt-get install python3-pandas
  $ sudo apt install ros-jazzy-rosidl-runtime-py


1.record ROS2 bag
 $ ros2 bag record -a
  
end record with control c

2. run script in goal directory of yaml and mcap file (adjust filename in code)
3. get csv file of every topic with timestamp in the first row
