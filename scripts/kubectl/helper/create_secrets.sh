#!/bin/bash

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a create_secrets.log
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    log "Error: kubectl is not installed. Please install kubectl and try again."
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

log "Creating secrets..."

# Create the general token (gitlab-token)
kubectl create secret generic gitlab-token \
  --from-literal=access-token=$ACCESS_TOKEN

# Create the Docker registry secret (gitlab-docker-secret)
kubectl create secret docker-registry gitlab-docker-secret \
  --docker-server=$DOCKER_REGISTRY_URL \
  --docker-username=$USERNAME \
  --docker-password=$ACCESS_TOKEN \
  --docker-email=$EMAIL

log "Secrets created successfully."

# Check if secrets have been added to your namespace
kubectl get secrets

log "Script execution completed."