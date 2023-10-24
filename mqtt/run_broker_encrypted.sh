# mqtt-broker encrypted
docker run \
    --rm \
    --network host \
    --name mosquitto-encrypted \
    -v $PWD/configs/mosquitto.ssl-all.conf:/mosquitto/config/mosquitto.conf:ro \
    -v $PWD/encryption/server:/mosquitto/encryption/server:ro \
    -v $PWD/encryption/certificate-authority:/mosquitto/encryption/certificate-authority:ro \
    --user "$(id -u):$(id -g)" \
    eclipse-mosquitto
