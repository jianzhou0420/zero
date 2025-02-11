PODS=$(kubectl get pod | grep $1 | awk '{print $1}')
kubectl exec -it $PODS -- /bin/bash