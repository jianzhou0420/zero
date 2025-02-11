if [ -z "$1" ]; then
  echo "Usage: $0 <pod-name>"
  PODS=$(kubectl get pod | grep '^t0-' | awk '{print $1}')
    kubectl exec -it $PODS -- /bin/bash
else
    kubectl exec -it $1 -- /bin/bash
    fi


