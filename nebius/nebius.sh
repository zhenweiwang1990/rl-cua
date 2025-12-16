#!/bin/bash
# shellcheck disable=SC2155

set -e -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd $SCRIPT_DIR

COMMON_SCRIPT_PATH="${SCRIPT_DIR}/deploy-vllm-common.sh"
if [[ -f "${COMMON_SCRIPT_PATH}" ]]; then
    # shellcheck source=deploy-vllm-common.sh
    source "${COMMON_SCRIPT_PATH}"
else
    echo "Error: Common deployment script not found at ${COMMON_SCRIPT_PATH}"
    echo "Deployment verification features will be unavailable."
fi

# Colors for output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Load environment variables from .env file if it exists
[[ -f ./.env ]] && source ./.env

# Check if nebius CLI is available
if ! command -v nebius &> /dev/null; then
    echo "Error: nebius CLI is not installed or not in PATH"
    echo "Please install it from https://docs.nebius.com/cli/install"
    exit 1
fi

# Function to get Nebius instances
function get_nebius_instances() {
    nebius compute instance list --format json 2>/dev/null || {
        echo "Error: Failed to get instances from Nebius CLI"
        exit 1
    }
}

# Function to start a Nebius instance
function start_nebius_instance() {
    local instance_id="$1"
    local instance_name="$2"
    
    echo "Starting instance $instance_name ($instance_id)..." >&2
    local start_output
    start_output=$(nebius compute instance start --id "$instance_id" 2>&1)
    local start_exit_code=$?
    
    if [[ $start_exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ Instance start command sent successfully${NC}" >&2
        return 0
    else
        echo "Error: Failed to start instance" >&2
        if [[ -n "$start_output" ]]; then
            echo "Error details: $start_output" >&2
        fi
        return 1
    fi
}

# Function to wait for instance to be running and get IP
function wait_for_instance_ready() {
    local instance_id="$1"
    local instance_name="$2"
    local max_wait_seconds="${3:-300}"  # Default 5 minutes
    local poll_interval_seconds="${4:-10}"  # Default 10 seconds
    
    echo "Waiting for instance to be ready..." >&2
    local start_time=$(date +%s)
    local attempt=1
    
    while true; do
        local instance_json=$(nebius compute instance get "$instance_id" --format json 2>/dev/null || echo "")
        
        if [[ -z "$instance_json" ]]; then
            echo "Warning: Could not get instance status (attempt $attempt)" >&2
        else
            local state=$(echo "$instance_json" | jq -r '.status.state // "UNKNOWN"')
            local ip=$(echo "$instance_json" | jq -r 'if .status.network_interfaces and (.status.network_interfaces | length > 0) and .status.network_interfaces[0].public_ip_address.address then .status.network_interfaces[0].public_ip_address.address else "" end' | sed 's|/.*||')
            
            if [[ "$state" == "RUNNING" ]] && [[ -n "$ip" ]] && [[ "$ip" != "null" ]]; then
                echo -e "${GREEN}✓ Instance is now RUNNING with IP: $ip${NC}" >&2
                echo "$ip"
                return 0
            fi
            
            local elapsed=$(( $(date +%s) - start_time ))
            if (( elapsed >= max_wait_seconds )); then
                echo "Timeout: Instance did not become ready within ${max_wait_seconds}s" >&2
                echo "Current state: $state" >&2
                return 1
            fi
            
            echo "Instance state: $state (attempt $attempt, elapsed: ${elapsed}s)..." >&2
        fi
        
        sleep "${poll_interval_seconds}"
        attempt=$((attempt + 1))
    done
}

# Function to parse SSH username from cloud_init_user_data
function parse_ssh_username() {
    local user_data="$1"
    
    if [[ -z "$user_data" ]] || [[ "$user_data" == "null" ]]; then
        echo "ubuntu"
        return
    fi
    
    # Convert escaped \n to actual newlines for easier parsing
    local normalized_data=$(echo "$user_data" | sed 's/\\n/\n/g')
    
    # Try to extract username from YAML format: " - name: username"
    # Method 1: Look for lines containing "name:" but not "ssh_authorized_keys"
    # Use awk to extract the field after "name:"
    local username=$(echo "$normalized_data" | grep "name:" | grep -v "ssh_authorized_keys" | head -n1 | awk -F'name:' '{print $2}' | awk '{print $1}' | tr -d '[:space:]' || echo "")
    
    # Clean up: remove any remaining dashes, colons, or other unwanted characters
    username=$(echo "$username" | sed 's/^-\+//' | sed 's/:$//' | sed 's/^-name://' | tr -d '[:space:]')
    
    # Default to ubuntu if parsing fails or result is empty/invalid
    if [[ -z "$username" ]] || [[ "$username" == "-name:" ]] || [[ "$username" == "name:" ]] || [[ "$username" == "-" ]]; then
        echo "ubuntu"
    else
        echo "$username"
    fi
}

# Function to display instances and let user select one
function select_instance() {
    local instances_json="$1"
    
    # Parse and display instances
    echo "" >&2
    echo "Available instances:" >&2
    
    local count=$(echo "$instances_json" | jq -r '.items | length' 2>/dev/null || echo "0")
    
    if [[ "$count" -eq 0 ]] || [[ "$count" == "null" ]]; then
        echo "No instances found." >&2
        return 1
    fi
    
    # Display instances with index
    echo "$instances_json" | jq -r '.items[] | 
        "\(.metadata.id)|\(.metadata.name)|\(.status.state // "UNKNOWN")|\(.spec.resources.platform // "unknown")|\(.spec.resources.preset // "unknown")|\(if .status.network_interfaces and (.status.network_interfaces | length > 0) and .status.network_interfaces[0].public_ip_address.address then .status.network_interfaces[0].public_ip_address.address else "N/A" end)|\(.spec.cloud_init_user_data // "")"' | \
    grep -v '^$' | \
    awk -F'|' 'BEGIN {
        printf "%-4s %-30s %-12s %-20s %-20s %-15s\n", "No.", "Name", "State", "Platform", "Preset", "IP";
        print "-------------------------------------------------------------------------------------------------------"
        counter = 0
    }
    NF >= 6 {
        counter++
        # Extract IP, handling cases where it might be empty or "null"
        ip = $6
        if (ip == "" || ip == "null") {
            ip = "N/A"
        } else if (ip != "N/A") {
            # Remove CIDR suffix (e.g., /32) from IP address
            sub(/\/[0-9]+$/, "", ip)
        }
        # Extract state
        state = $3
        if (state == "" || state == "null") {
            state = "UNKNOWN"
        }
        printf "%-4s %-30s %-12s %-20s %-20s %-15s\n", counter".", $2, state, $4, $5, ip
    }' >&2
    
    echo "" >&2
    read -p "Select instance number (1-$count, or 0 to cancel): " selection
    
    if [[ "$selection" -eq 0 ]]; then
        echo "Cancelled." >&2
        return 1
    fi
    
    if [[ "$selection" -lt 1 ]] || [[ "$selection" -gt "$count" ]]; then
        echo "Invalid selection." >&2
        return 1
    fi
    
    # Get selected instance info (0-indexed)
    local idx=$((selection - 1))
    local selected_id=$(echo "$instances_json" | jq -r ".items[$idx].metadata.id")
    local selected_name=$(echo "$instances_json" | jq -r ".items[$idx].metadata.name")
    local selected_state=$(echo "$instances_json" | jq -r ".items[$idx].status.state // \"UNKNOWN\"")
    # Try to get IP from status.network_interfaces, handle cases where public_ip_address might be empty object {}
    # Also remove CIDR suffix (e.g., /32) from IP address
    local selected_ip=$(echo "$instances_json" | jq -r ".items[$idx] | if .status.network_interfaces and (.status.network_interfaces | length > 0) and .status.network_interfaces[0].public_ip_address.address then .status.network_interfaces[0].public_ip_address.address else \"\" end" | sed 's|/.*||')
    local selected_platform=$(echo "$instances_json" | jq -r ".items[$idx].spec.resources.platform // \"unknown\"")
    local selected_preset=$(echo "$instances_json" | jq -r ".items[$idx].spec.resources.preset // \"unknown\"")
    local user_data=$(echo "$instances_json" | jq -r ".items[$idx].spec.cloud_init_user_data // \"\"")
    local ssh_username=$(parse_ssh_username "$user_data")
    
    # Normalize IP - use "N/A" if empty or null
    if [[ -z "$selected_ip" ]] || [[ "$selected_ip" == "null" ]]; then
        selected_ip="N/A"
    fi
    
    # Normalize state
    if [[ -z "$selected_state" ]] || [[ "$selected_state" == "null" ]]; then
        selected_state="UNKNOWN"
    fi
    
    echo "" >&2
    echo "Selected instance: $selected_name" >&2
    if [[ "$selected_ip" != "N/A" ]]; then
        echo "IP: $selected_ip" >&2
    else
        echo "IP: N/A (instance may be stopped or starting)" >&2
    fi
    echo "State: $selected_state" >&2
    echo "Platform: $selected_platform" >&2
    echo "Preset: $selected_preset" >&2
    echo "SSH username: $ssh_username" >&2
    echo "$selected_id|$selected_name|$selected_ip|$selected_platform|$selected_preset|$ssh_username"
}

# Function to create HTTP proxy script
function create_proxy_script() {
    local proxy_host="$1"
    local proxy_port="$2"
    local proxy_script=$(mktemp)
    
    cat > "$proxy_script" << 'EOF'
#!/bin/bash
# Send HTTP CONNECT request and read response headers, then relay data
exec 3<>/dev/tcp/$3/$4
printf "CONNECT $1:$2 HTTP/1.0\r\n\r\n" >&3
while read -r line <&3; do
    line=$(echo "$line" | tr -d '\r')
    [[ -z "$line" ]] && break
done
cat <&3 & cat >&3
EOF
    chmod +x "$proxy_script"
    echo "$proxy_script"
}

# Function to get proxy command options for SSH/SCP
function get_proxy_options() {
    if [[ -n "$http_proxy" ]]; then
        local proxy_host=$(echo "$http_proxy" | sed -E 's|https?://||' | cut -d: -f1)
        local proxy_port=$(echo "$http_proxy" | sed -E 's|https?://||' | cut -d: -f2)
        local proxy_script=$(create_proxy_script "$proxy_host" "$proxy_port")
        echo "-o ProxyCommand=\"$proxy_script %h %p ${proxy_host} ${proxy_port}\""
    fi
}

# Function to run SCP with optional proxy support
function scp_with_proxy() {
    local source="$1"
    local dest="$2"
    local ssh_user="$3"
    local proxy_opts=$(get_proxy_options)
    
    if [[ -n "$proxy_opts" ]]; then
        eval command scp $proxy_opts "$source" "$ssh_user@$dest"
    else
        command scp "$source" "$ssh_user@$dest"
    fi
}

# Function to run SSH with optional proxy support
function ssh_with_proxy() {
    local ip="$1"
    local ssh_user="$2"
    local ssh_cmd="$3"
    local proxy_opts=$(get_proxy_options)
    
    if [[ -n "$proxy_opts" ]]; then
        eval exec command ssh $proxy_opts -t "$ssh_user@$ip" "$ssh_cmd"
    else
        exec command ssh -t "$ssh_user@$ip" "$ssh_cmd"
    fi
}

# Function to connect via SSH with optional proxy (for interactive login)
function ssh_connect() {
    local ip="$1"
    local ssh_user="$2"
    local proxy_opts=$(get_proxy_options)
    
    # Add connection timeout and other useful SSH options
    local ssh_opts="-o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no"
    
    if [[ -n "$proxy_opts" ]]; then
        # Use eval to properly expand proxy options
        # Note: exec replaces the current process, so script ends here
        eval exec command ssh $ssh_opts $proxy_opts "$ssh_user@$ip"
    else
        exec command ssh $ssh_opts "$ssh_user@$ip"
    fi
}

# Function to run a remote command and capture output
function ssh_capture() {
    local ip="$1"
    local ssh_user="$2"
    local remote_cmd="$3"
    local proxy_opts
    proxy_opts=$(get_proxy_options)

    local remote_cmd_escaped
    printf -v remote_cmd_escaped '%q' "${remote_cmd}"

    if [[ -n "${proxy_opts}" ]]; then
        eval command ssh ${proxy_opts} "$ssh_user@${ip}" bash -lc "${remote_cmd_escaped}"
    else
        command ssh "$ssh_user@${ip}" bash -lc "${remote_cmd_escaped}"
    fi
}

function ssh_docker_ps_json() {
    local ip="$1"
    local ssh_user="$2"
    local proxy_opts
    proxy_opts=$(get_proxy_options)

    if [[ -n "${proxy_opts}" ]]; then
        eval command ssh ${proxy_opts} "$ssh_user@${ip}" docker ps --filter label=vllm-model --format json
    else
        command ssh "$ssh_user@${ip}" docker ps --filter label=vllm-model --format json
    fi
}

# Function to select vLLM image based on platform
function select_image_by_platform() {
    local platform="$1"
    
    # Convert platform to lowercase for comparison
    local platform_lower=$(echo "$platform" | tr '[:upper:]' '[:lower:]')
    
    if echo "$platform_lower" | grep -q "h100"; then
        echo "vllm/vllm-openai:latest"
    elif echo "$platform_lower" | grep -q "h200"; then
        echo "vllm/vllm-openai:latest"
    elif echo "$platform_lower" | grep -q "gh200\|grace.hopper"; then
        echo "rajesh550/gh200-vllm:0.11.1rc2"
    else
        # Unknown type, return empty to let script auto-detect
        echo ""
    fi
}

# Function to run a remote script (optionally with vLLM environment)
function run_remote_script() {
    local instance_name="$1"
    local instance_ip="$2"
    local ssh_user="$3"
    local script_name="$4"
    local target_label="$5"
    local requires_vllm_env="${6:-false}"
    local platform="${7:-}"  # Optional platform for vLLM image selection
    local deploy_script_path="$SCRIPT_DIR/$script_name"
    local common_script_path="$SCRIPT_DIR/deploy-vllm-common.sh"

    if [[ ! -f "$deploy_script_path" ]]; then
        echo "Error: Deployment script not found at $deploy_script_path"
        exit 1
    fi

    if [[ "${requires_vllm_env}" == "true" ]]; then
        if [[ -z "${VLLM_API_KEY}" ]]; then
            echo "Error: VLLM_API_KEY is not set"
            echo "Please add VLLM_API_KEY to your .env file"
            echo "See env.example for reference"
            exit 1
        fi
        if [[ ! -f "$common_script_path" ]]; then
            echo "Error: Common deployment script not found at $common_script_path"
            exit 1
        fi
    fi

    echo ""
    echo "Deploying $target_label to $instance_name ($instance_ip)..."
    if [[ -n "${platform}" ]]; then
        echo "Platform: $platform"
    fi
    echo "Uploading deployment scripts..."

    scp_with_proxy "$deploy_script_path" "$instance_ip:/tmp/$script_name" "$ssh_user"
    if [[ "${requires_vllm_env}" == "true" ]]; then
        scp_with_proxy "$common_script_path" "$instance_ip:/tmp/deploy-vllm-common.sh" "$ssh_user"
    fi

    local remote_cmd=""
    if [[ "${requires_vllm_env}" == "true" ]]; then
        printf -v remote_env "VLLM_API_KEY=%q" "${VLLM_API_KEY}"
        
        # Select vLLM image based on platform if available
        if [[ -n "${platform}" ]]; then
            local vllm_image=$(select_image_by_platform "$platform")
            if [[ -n "${vllm_image}" ]]; then
                printf -v image_env "VLLM_IMAGE=%q" "${vllm_image}"
                remote_env="${remote_env} ${image_env}"
                echo "Selected vLLM image based on platform: $vllm_image"
            fi
        fi
        
        if [[ -n "${HOST_PORT:-}" ]]; then
            printf -v host_port_env "HOST_PORT=%q" "${HOST_PORT}"
            remote_env="${remote_env} ${host_port_env}"
        fi
        echo "Using VLLM_API_KEY from local environment..."
        remote_cmd="${remote_env} bash /tmp/${script_name} && rm -f /tmp/${script_name} /tmp/deploy-vllm-common.sh"
    else
        remote_cmd="bash /tmp/${script_name} && rm -f /tmp/${script_name}"
    fi

    ssh_with_proxy "$instance_ip" "$ssh_user" "${remote_cmd}"
}

# ==============================================================================
# Main Script
# ==============================================================================

# Step 1: Get instances
echo "Fetching instances from Nebius..."
instances=$(get_nebius_instances)

# Step 2: Select instance
selected_info=$(select_instance "$instances")
if [[ $? -ne 0 ]] || [[ -z "$selected_info" ]]; then
    exit 1
fi

IFS='|' read -r instance_id instance_name instance_ip platform preset ssh_username <<< "$selected_info"

# Step 2.5: Check if instance is STOPPED and ask to start it
# Get current state from the instances data we already have
current_instance=$(echo "$instances" | jq -r --arg id "$instance_id" '.items[] | select(.metadata.id == $id)')
current_state=$(echo "$current_instance" | jq -r '.status.state // "UNKNOWN"')

if [[ "$current_state" == "STOPPED" ]]; then
    if [[ "$instance_ip" == "N/A" ]] || [[ -z "$instance_ip" ]]; then
        echo ""
        echo "Instance $instance_name is currently STOPPED."
        read -p "Do you want to start it? (Y/n): " start_choice
        start_choice=${start_choice:-y}
        
        if [[ "$start_choice" == "y" ]] || [[ "$start_choice" == "Y" ]]; then
            if start_nebius_instance "$instance_id" "$instance_name"; then
                echo ""
                echo "Waiting for instance to start and get IP address..."
                instance_ip=$(wait_for_instance_ready "$instance_id" "$instance_name")
                
                if [[ -z "$instance_ip" ]] || [[ "$instance_ip" == "null" ]]; then
                    echo "Error: Failed to get IP address after starting instance" >&2
                    echo "You may need to wait a bit longer and try again." >&2
                    exit 1
                fi
                
                echo ""
                echo "Instance is ready! IP: $instance_ip"
                echo ""
            else
                echo "Failed to start instance. Exiting." >&2
                exit 1
            fi
        else
            echo "Instance is stopped. Some operations may not be available." >&2
        fi
    fi
fi

# Step 3: Select operation and execute
echo ""
echo "Select operation:"
echo "  1) SSH login"
echo "  2) Deploy Gelato-30B-A3B"
echo "  3) Verify vLLM deployment"
echo "  4) Cancel"
echo ""
read -p "Enter your choice (1-4, default[1]): " operation

operation=${operation:-1}

case $operation in
    1)
        if [[ "$instance_ip" == "N/A" ]] || [[ -z "$instance_ip" ]]; then
            echo "Error: Cannot connect via SSH - instance does not have a public IP address" >&2
            echo "Instance state may be STOPPED. Please start the instance first." >&2
            exit 1
        fi
        echo ""
        echo "Connecting to $instance_name ($instance_ip) as $ssh_username..."
        echo "(You may be prompted for SSH key passphrase or password)"
        echo ""
        ssh_connect "$instance_ip" "$ssh_username"
        ;;
    2)
        if [[ "$instance_ip" == "N/A" ]] || [[ -z "$instance_ip" ]]; then
            echo "Error: Cannot deploy - instance does not have a public IP address" >&2
            echo "Instance state may be STOPPED. Please start the instance first." >&2
            exit 1
        fi
        run_remote_script "$instance_name" "$instance_ip" "$ssh_username" "deploy-gelato-30b-a3b.sh" "Gelato-30B-A3B" "true" "$platform"
        ;;
    3)
        if [[ "$instance_ip" == "N/A" ]] || [[ -z "$instance_ip" ]]; then
            echo "Error: Cannot verify - instance does not have a public IP address" >&2
            echo "Instance state may be STOPPED. Please start the instance first." >&2
            exit 1
        fi
        "${SCRIPT_DIR}/verify-vllm.sh" "$instance_ip" "$ssh_username"
        ;;
    4)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

