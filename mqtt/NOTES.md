## new docker run notes


### mosquitto broker setup

```
docker run -it -p 1883:1883 -p 9001:9001 --network host eclipse-mosquitto
docker run -it --network host eclipse-mosquitto
```

--network host => for same system
else use overlay


#### broker encryption setup

ref:
https://github.com/ika-rwth-aachen/mqtt-in-docker#implementing-a-public-key-infrastructure-using-openssl



>> trials

```
  rosrun mqtt_node cloud_node.py -m src/mqtt_node/src/model/mobilenet_v3_large_968_608_os8.pb 

  rosrun mqtt_node cloud_node.py -m model/mobilenet_v3_large_968_608_os8.pb 
  rosrun mqtt_node cloud_node.py -m model/mobilenetv3_large_os8_deeplabv3plus_72miou -x xml/cityscapes.xml -s
  rosrun mqtt_node cloud_node.py -m model/best_weights_e=00231_val_loss=0.1518/ -x xml/cityscapes.xml -s
  
```



### vehicle and cloud nodes setup


1. move to git directory
    ```
    cd vehicle-cloud-inference
    ```

2. setup ros workspaces (###first time setup)
    ```
    cd ..
    ./setup_new_ros_ws.sh
    ```
    note: run as normal user , not docker

3. start docker containers in separate terminals

    vehicle node start docker
    ```
    ./docker/run.sh -n cloud_node -a
    ```

    cloud node start docker
    ```
    ./docker/run.sh -n cloud_node -a
    ```


4. switch to created 'rosuser' from default 'root' user
    - needed as 'rwthika/acdc:latest' docker image starts with root user and causes issues

    ```
    ../user_setup_docker.sh
    ```

    or directly,
    ```
    su rosuser
    # password: rosuser
    ```

5. switch to workspace and source devel in shell
    
    cloud_node docker
    ```
    cd ws_mqtt_nodes/cloud_node
    catkin_make
    source devel/setup.bash
    ```

    vehicle_node docker
    ```
    cd ws_mqtt_nodes/vehicle_node
    catkin_make
    source devel/setup.bash
    ```

  6. start the nodes

  > cloud node
    ```
    roslaunch mqtt_node cloud_node.launch
    ```
  > vehicle node (optional with rosbag)
    ```
    roslaunch mqtt_node vehicle_node.launch
    ```
    ```
    roslaunch mqtt_node vehicle_node_with_rosbag.launch
    ```
  > rosbag separate terminal; 
    - also needs steps 3 and 4 in new terminal
    ```
    roslaunch mqtt_node run_rosbag.launch
    ```

instead of 5-6, can also do

  > cloud node
    ```
    source ../start_node.sh cloud_node
    ```
  > vehicle node (optional with rosbag)
    ```
    source ../start_node.sh vehicle_node
    ```
  > rosbag separate terminal; 
    - also needs steps 3 and 4 in new terminal, for vehicle node
    ```
    cd ws_mqtt_nodes/vehicle_node
    catkin_make
    source devel/setup.bash
    roslaunch mqtt_node run_rosbag.launch
    ```
