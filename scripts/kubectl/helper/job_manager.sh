#!/bin/bash

# Function to get the pod name for a job
get_pod_name() {
    local job_name=$1
    local namespace=$2
    kubectl get pods -n "$namespace" --selector=job-name="$job_name" -o jsonpath='{.items[0].metadata.name}'
}

# Function to describe the job status
describe_job_status() {
    local job=$1
    local namespace=$2
    echo "Status for job: $job"
    kubectl get job "$job" -n "$namespace" -o jsonpath='{range .status.conditions[*]}{.type}{"\t"}{.status}{"\t"}{.lastTransitionTime}{"\t"}{.reason}{"\t"}{.message}{"\n"}{end}' | 
    while IFS=$'\t' read -r type status lastTransitionTime reason message; do
        printf "%-15s %-10s %-30s %-20s %s\n" "$type" "$status" "$lastTransitionTime" "$reason" "$message"
    done
    echo ""
}

# Function to view job logs
view_job_logs() {
    local job_name=$1
    local namespace=$2
    local pod_name=$(get_pod_name "$job_name" "$namespace")
    if [ -z "$pod_name" ]; then
        echo "No pod found for job $job_name"
        return
    fi
    echo "Viewing logs for job $job_name (pod: $pod_name):"
    kubectl logs -n "$namespace" "$pod_name"
}

# Function to exec into pod
exec_into_pod() {
    local job_name=$1
    local namespace=$2
    local pod_name=$(get_pod_name "$job_name" "$namespace")
    if [ -z "$pod_name" ]; then
        echo "No pod found for job $job_name"
        return
    fi
    echo "Executing into pod $pod_name for job $job_name"
    echo "Note: To exit the pod and return to this menu, type 'exit' in the pod's shell."
    kubectl exec -it -n "$namespace" "$pod_name" -- /bin/bash
}

# Function to list all jobs and their statuses
list_all_jobs() {
    local namespace=$1
    local jobs=($(kubectl get jobs -n "$namespace" -o jsonpath='{.items[*].metadata.name}'))
    if [ ${#jobs[@]} -eq 0 ]; then
        echo "No jobs found in namespace $namespace"
        return
    fi
    printf "%-20s %-15s %-10s %-30s %-20s %s\n" "Job Name" "Condition Type" "Status" "Last Transition Time" "Reason" "Message"
    echo "---------------------------------------------------------------------------------------------------------------------"
    for job in "${jobs[@]}"; do
        printf "%-20s " "$job"
        describe_job_status "$job" "$namespace"
    done
}

# Main script
echo "Kubernetes Job Manager"
echo "----------------------"

# Get the current namespace
NAMESPACE=$(kubectl config view --minify --output 'jsonpath={..namespace}')
if [ -z "$NAMESPACE" ]; then
    read -p "No namespace is set in the current context. Please enter a namespace: " NAMESPACE
    if [ -z "$NAMESPACE" ]; then
        echo "No namespace entered. Exiting."
        exit 1
    fi
fi
echo "Using namespace: $NAMESPACE"

# Function to list jobs and let user select one
select_job() {
    local jobs=($(kubectl get jobs -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'))
    if [ ${#jobs[@]} -eq 0 ]; then
        echo "No jobs found in namespace $NAMESPACE"
        return
    fi
    echo "Jobs in namespace $NAMESPACE:"
    for i in "${!jobs[@]}"; do
        echo "$((i+1))) ${jobs[i]}"
    done
    while true; do
        read -p "Enter the number of the job you want to manage (1-${#jobs[@]}, or 0 to cancel): " selection
        if [ "$selection" -eq 0 ]; then
            return
        elif [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#jobs[@]}" ]; then
            JOB_NAME="${jobs[$((selection-1))]}"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
}

# Main menu
while true; do
    echo ""
    echo "Select an action:"
    echo "1) List all jobs and their statuses"
    echo "2) Select a job to manage"
    echo "3) Exit"
    read -p "Enter your choice (1-3): " main_choice

    case $main_choice in
        1) list_all_jobs "$NAMESPACE" ;;
        2) 
            select_job
            if [ -n "$JOB_NAME" ]; then
                while true; do
                    echo ""
                    echo "Selected Job: $JOB_NAME"
                    echo "Select an action:"
                    echo "1) View job status"
                    echo "2) View job logs"
                    echo "3) Exec into pod"
                    echo "4) Back to main menu"
                    read -p "Enter your choice (1-4): " job_choice

                    case $job_choice in
                        1) describe_job_status "$JOB_NAME" "$NAMESPACE" ;;
                        2) view_job_logs "$JOB_NAME" "$NAMESPACE" ;;
                        3) 
                           echo "Note: After exec-ing into the pod, type 'exit' to return to this menu."
                           exec_into_pod "$JOB_NAME" "$NAMESPACE" 
                           ;;
                        4) break ;;
                        *) echo "Invalid choice. Please try again." ;;
                    esac
                done
            fi
            ;;
        3) echo "Exiting."; exit 0 ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
done