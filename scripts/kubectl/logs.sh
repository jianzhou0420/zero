PODS=$(kubectl get pod | grep $1 | awk '{print $1}')
kubectl logs -f $PODS
# echo $PODS