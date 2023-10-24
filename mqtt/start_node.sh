NODE=$1

cd $NODE
catkin_make
. devel/setup.bash

roslaunch mqtt_node $NODE.launch