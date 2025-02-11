#!/bin/bash

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a docker_login.log
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log "Error: Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if sufficient arguments are provided
if [ "$#" -ne 2 ]; then
    log "Usage: $0 ACCESS_TOKEN EMAIL"
    exit 1
fi

# Assign variables from input arguments
ACCESS_TOKEN=$1
EMAIL=$2
DOCKER_REGISTRY_URL=docker.aiml.team

# Extract username from email
USERNAME="${EMAIL%@*}"

# Perform docker login with sudo
log "Attempting to log in to Docker registry..."
if sudo docker login $DOCKER_REGISTRY_URL --username $USERNAME --password $ACCESS_TOKEN; then
    log "Docker login successful."
else
    log "Docker login failed."
    exit 1
fi

log "Script execution completed."