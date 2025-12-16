#!/bin/bash
set -euo pipefail

print_header() {
    local title="$1"
    echo ""
    echo "=========================================="
    echo "${title}"
    echo "=========================================="
}

detect_gpu_type() {
    # Detect GPU type using nvidia-smi
    # Returns: "h100", "gh200", or "unknown"
    if ! command -v nvidia-smi &> /dev/null; then
        echo "unknown"
        return 1
    fi
    
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || echo "")
    
    if [[ -z "${gpu_name}" ]]; then
        echo "unknown"
        return 1
    fi
    
    if echo "${gpu_name}" | grep -q "h100"; then
        echo "h100"
        return 0
    elif echo "${gpu_name}" | grep -q "gh200\|grace hopper"; then
        echo "gh200"
        return 0
    else
        echo "unknown"
        return 1
    fi
}

select_vllm_image() {
    # Select vLLM image based on GPU type
    # Usage: select_vllm_image [override_image]
    # If override_image is provided, use it; otherwise auto-detect based on GPU type
    local override_image="${1:-}"
    
    if [[ -n "${override_image}" ]]; then
        echo "${override_image}"
        return 0
    fi
    
    local gpu_type
    gpu_type=$(detect_gpu_type)
    
    case "${gpu_type}" in
        h100)
            echo "vllm/vllm-openai:latest"
            ;;
        gh200)
            echo "rajesh550/gh200-vllm:0.11.1rc2"
            ;;
        *)
            echo "Error: Unable to detect GPU type. Please specify VLLM_IMAGE manually." >&2
            return 1
            ;;
    esac
}

install_nvidia_container_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo "✓ NVIDIA Container Toolkit is already installed"
        nvidia-ctk --version
        return
    fi

    echo "Installing prerequisites..."
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        curl \
        gnupg2

    echo "Configuring NVIDIA Container Toolkit repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    echo "Updating package list..."
    sudo apt-get update

    echo "Installing NVIDIA Container Toolkit..."
    local toolkit_version="1.18.0-1"
    sudo apt-get install -y \
        "nvidia-container-toolkit=${toolkit_version}" \
        "nvidia-container-toolkit-base=${toolkit_version}" \
        "libnvidia-container-tools=${toolkit_version}" \
        "libnvidia-container1=${toolkit_version}"

    echo "✓ NVIDIA Container Toolkit installed successfully"
}

install_docker() {
    # Check if Docker is already installed and working
    if command -v docker >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
        echo "✓ Docker is already installed"
        sudo docker --version
        return
    fi

    echo "Docker is not installed. Installing Docker..."
    
    # Remove old versions if any
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Install prerequisites
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin
    
    # Start and enable Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add current user to docker group
    if ! id -nG "${USER}" | tr ' ' '\n' | grep -q '^docker$'; then
        sudo usermod -aG docker "${USER}"
        echo "Added ${USER} to docker group"
    fi
    
    # Verify installation
    if sudo docker info >/dev/null 2>&1; then
        echo "✓ Docker installed successfully"
        sudo docker --version
    else
        echo "✗ Docker installation verification failed"
        return 1
    fi
}

restart_docker() {
    # Detect Docker installation method and restart accordingly
    # Returns 0 on success, 1 on failure
    
    # Check if Docker is installed via snap
    if command -v snap >/dev/null 2>&1 && snap list 2>/dev/null | grep -q "^docker\s"; then
        echo "Detected Docker installed via snap, restarting..."
        if sudo snap restart docker; then
            echo "✓ Docker restarted successfully via snap"
            return 0
        else
            echo "Warning: Failed to restart Docker via snap"
            return 1
        fi
    fi
    
    # Check for systemd service (try common service names)
    local docker_service=""
    for service_name in docker docker.io containerd; do
        if systemctl list-units --type=service --all 2>/dev/null | grep -q "^${service_name}\.service"; then
            docker_service="${service_name}"
            break
        fi
    done
    
    if [[ -n "${docker_service}" ]]; then
        echo "Detected Docker systemd service: ${docker_service}, restarting..."
        if sudo systemctl restart "${docker_service}"; then
            echo "✓ Docker restarted successfully via systemctl"
            return 0
        else
            echo "Warning: Failed to restart Docker via systemctl"
            return 1
        fi
    fi
    
    # If no service found, try to restart Docker daemon directly
    echo "Warning: Could not detect Docker installation method"
    echo "Attempting to restart Docker daemon..."
    
    # Try systemctl with docker (might work even if service not listed)
    if sudo systemctl restart docker 2>/dev/null; then
        echo "✓ Docker restarted successfully"
        return 0
    fi
    
    # Last resort: try to kill and let it restart (if using systemd)
    if pgrep dockerd >/dev/null 2>&1 || pgrep -f "docker daemon" >/dev/null 2>&1; then
        echo "Warning: Docker process found but could not restart via service manager"
        echo "Please manually restart Docker or check the installation"
        return 1
    fi
    
    echo "Warning: Could not find Docker daemon process"
    return 1
}

diagnose_docker_nvidia() {
    echo ""
    echo "=== Docker and NVIDIA Runtime Diagnostics ==="
    
    # Check Docker status
    echo ""
    echo "1. Docker Status:"
    if command -v docker >/dev/null 2>&1; then
        echo "   Docker version: $(docker --version 2>&1 || echo 'N/A')"
        if sudo docker info >/dev/null 2>&1; then
            echo "   ✓ Docker daemon is running"
        else
            echo "   ✗ Docker daemon is not running or not accessible"
            return 1
        fi
    else
        echo "   ✗ Docker is not installed"
        return 1
    fi
    
    # Check NVIDIA runtime
    echo ""
    echo "2. Docker Runtimes:"
    local runtimes
    runtimes=$(sudo docker info --format '{{range $name, $_ := .Runtimes}}{{printf "%s " $name}}{{end}}' 2>/dev/null || echo "")
    if [[ -n "${runtimes}" ]]; then
        echo "   Available runtimes: ${runtimes}"
        if echo "${runtimes}" | grep -qw "nvidia"; then
            echo "   ✓ NVIDIA runtime is configured"
        else
            echo "   ✗ NVIDIA runtime is NOT configured"
        fi
    else
        echo "   ✗ Could not retrieve runtime information"
    fi
    
    # Check Docker daemon.json
    echo ""
    echo "3. Docker Daemon Configuration (/etc/docker/daemon.json):"
    local docker_config="/etc/docker/daemon.json"
    if [[ -f "${docker_config}" ]]; then
        echo "   Configuration file exists:"
        sudo cat "${docker_config}" | sed 's/^/   /' || echo "   (could not read)"
        
        if sudo grep -q '"cdi"' "${docker_config}" 2>/dev/null; then
            echo "   ⚠ Warning: CDI configuration detected (may conflict with --runtime nvidia)"
        fi
    else
        echo "   Configuration file does not exist (using defaults)"
    fi
    
    # Check NVIDIA Container Toolkit
    echo ""
    echo "4. NVIDIA Container Toolkit:"
    if command -v nvidia-ctk >/dev/null 2>&1; then
        echo "   ✓ nvidia-ctk is installed"
        nvidia-ctk --version 2>&1 | sed 's/^/   /' || true
        
        # Check CDI
        echo ""
        echo "   CDI Status:"
        if sudo nvidia-ctk cdi list 2>/dev/null | grep -q "nvidia.com/gpu"; then
            echo "   ⚠ CDI devices are configured (may conflict with --runtime nvidia)"
            echo "   CDI devices:"
            sudo nvidia-ctk cdi list 2>/dev/null | grep "nvidia.com/gpu" | sed 's/^/     /' || true
        else
            echo "   ✓ No CDI devices configured (good for --runtime nvidia)"
        fi
    else
        echo "   ✗ nvidia-ctk is not installed"
    fi
    
    # Check nvidia-smi
    echo ""
    echo "5. NVIDIA Driver:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "   ✓ nvidia-smi is available"
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "")
        if [[ -n "${gpu_info}" ]]; then
            echo "   GPU: ${gpu_info}"
        fi
    else
        echo "   ✗ nvidia-smi is not available"
    fi
    
    # Test NVIDIA runtime
    echo ""
    echo "6. Testing NVIDIA Runtime:"
    if sudo docker run --rm --runtime nvidia nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        echo "   ✓ NVIDIA runtime test passed"
    else
        echo "   ✗ NVIDIA runtime test failed"
        echo "   This indicates --runtime nvidia may not work properly"
    fi
    
    echo ""
    echo "=== End of Diagnostics ==="
    echo ""
}

configure_docker_runtime() {
    # Check if NVIDIA runtime is already configured
    local has_nvidia_runtime=false
    if sudo docker info --format '{{range $name, $_ := .Runtimes}}{{printf "%s " $name}}{{end}}' 2>/dev/null | grep -qw "nvidia"; then
        has_nvidia_runtime=true
        echo "✓ Docker is already configured with NVIDIA runtime"
    fi

    # If NVIDIA runtime is not configured, set it up
    if [[ "${has_nvidia_runtime}" == "false" ]]; then
        echo "Configuring Docker to use NVIDIA runtime..."
        # Configure NVIDIA runtime without CDI
        sudo nvidia-ctk runtime configure --runtime=docker

        echo "Restarting Docker..."
        if ! restart_docker; then
            echo "Warning: Docker restart may have failed, but continuing..."
            echo "Please verify Docker is running and NVIDIA runtime is configured:"
            echo "  sudo docker info | grep -i runtime"
        fi
        
        # Verify runtime is configured
        if sudo docker info --format '{{range $name, $_ := .Runtimes}}{{printf "%s " $name}}{{end}}' 2>/dev/null | grep -qw "nvidia"; then
            echo "✓ NVIDIA runtime configured successfully"
        else
            echo "Warning: NVIDIA runtime may not be properly configured"
            echo "Running diagnostics again..."
            diagnose_docker_nvidia
        fi
    fi

    if id -nG "${USER}" | tr ' ' '\n' | grep -q '^docker$'; then
        echo "✓ Current user ${USER} already belongs to docker group"
    else
        echo "Adding ${USER} to docker group..."
        sudo usermod -aG docker "${USER}"
        echo "Switching current shell to docker group..."
        if command -v newgrp >/dev/null 2>&1; then
            newgrp docker <<'EOF'
echo "✓ Switched to docker group for current session"
EOF
        else
            echo "Warning: 'newgrp' command not found; log out and back in to refresh group membership."
        fi
    fi

    echo "✓ Docker runtime configured successfully"
}

require_vllm_api_key() {
    if [[ -z "${VLLM_API_KEY:-}" ]]; then
        echo "Error: VLLM_API_KEY environment variable is not set"
        exit 1
    fi
}

verify_vllm_deployment() {
    local host_port="$1"
    local model="$2"
    local api_key="$3"

    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required for deployment verification"
        return 1
    fi

    local host="${VERIFY_HOST:-127.0.0.1}"
    local prompt_text="${VERIFY_PROMPT_TEXT:-Click the ollama icon and return XY coordinates directly, similar to X,Y}"
    local image_url="${VERIFY_IMAGE_URL:-https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_092343_gen.png}"
    local max_wait_seconds="${VERIFY_MAX_WAIT_SECONDS:-600}"
    local poll_interval_seconds="${VERIFY_POLL_INTERVAL_SECONDS:-10}"
    local endpoint="http://${host}:${host_port}/v1/chat/completions"
    local payload
    payload=$(cat <<EOF
{
    "model": "${model}",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "${prompt_text}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "${image_url}"
                    }
                }
            ]
        }
    ]
}
EOF
)

    echo "Waiting for vLLM endpoint ${endpoint} to become ready (timeout ${max_wait_seconds}s)..."
    local start_time
    start_time=$(date +%s)
    local attempt=1
    local response=""
    local curl_output=""

    while true; do
        if curl_output=$(curl --silent --fail --max-time 60 \
            --header "Content-Type: application/json" \
            --header "Authorization: Bearer ${api_key}" \
            --data "${payload}" \
            "${endpoint}" 2>&1); then
            response="${curl_output}"
            break
        fi

        local elapsed=$(( $(date +%s) - start_time ))
        if (( elapsed >= max_wait_seconds )); then
            echo "✗ Deployment verification failed after ${max_wait_seconds}s (last error: ${curl_output})"
            return 1
        fi

        echo "Endpoint not ready yet (attempt ${attempt}), retrying in ${poll_interval_seconds}s..."
        sleep "${poll_interval_seconds}"
        attempt=$((attempt + 1))
    done

    echo -e "\033[1;32m✓\033[0m Endpoint is responding. Processing verification response..."

    echo "Verification response:"
    echo "==========================================="
    echo "${response}"
    echo "==========================================="
    echo ""

    if echo "${response}" | grep -q '"choices"'; then
        echo -e "\033[1;32m✓\033[0m Deployment verification succeeded"
        return 0
    fi

    echo "✗ Deployment verification failed: unexpected response format"
    return 1
}

container_is_running() {
    local container_name="$1"
    sudo docker ps --filter "name=^/${container_name}$" --format '{{.Names}}' | grep -q "^${container_name}$"
}

get_container_config_hash() {
    local container_name="$1"
    local label=""
    label=$(sudo docker inspect --format '{{ index .Config.Labels "vllm-config-hash" }}' "${container_name}" 2>/dev/null || true)
    echo "${label}"
}

remove_stopped_container() {
    local container_name="$1"
    local existing_id
    existing_id=$(sudo docker ps -a --filter "name=^/${container_name}$" --format '{{.ID}}')

    if [[ -n "${existing_id}" ]]; then
        echo "Found stopped container, removing it..."
        sudo docker rm "${container_name}" || true
    fi
}

show_container_status() {
    local container_name="$1"
    echo ""
    echo "Container status:"
    sudo docker ps -f "name=${container_name}"
}

deploy_vllm_container() {
    local container_name="$1"
    local image="$2"
    local model="$3"
    local host_port="$4"
    local entrypoint_string="$5"
    local command_string="$6"
    local envs_string="${7:-}"  # Optional 7th parameter for environment variables
    shift 7
    local model_args=("$@")
    
    # Parse environment variables from envs_string (format: "KEY1=VALUE1 KEY2=VALUE2" or empty)
    # Multiple env vars are separated by spaces
    local additional_env_vars=()
    if [[ -n "${envs_string}" ]]; then
        # Split by space and collect environment variables
        # Use readarray to properly handle spaces within values (though values with spaces should be quoted)
        readarray -t env_array <<< "${envs_string// /$'\n'}"
        for env_var in "${env_array[@]}"; do
            [[ -z "${env_var}" ]] && continue
            additional_env_vars+=("${env_var}")
        done
    fi

    print_header "Step 1: Installing Docker"
    install_docker

    print_header "Step 2: Installing NVIDIA Container Toolkit"
    install_nvidia_container_toolkit

    print_header "Step 3: Configuring Docker Runtime"
    configure_docker_runtime

    print_header "Step 4: Starting vLLM Container"
    require_vllm_api_key

    local logging_level="${VLLM_LOGGING_LEVEL:-DEBUG}"
    
    local api_key_hash
    api_key_hash=$(printf '%s' "${VLLM_API_KEY}" | sha256sum | awk '{print $1}')
    # Include envs_string in config hash to ensure container is recreated when env vars change
    local config_inputs=("${image}" "${model}" "${host_port}" "${logging_level}" "${entrypoint_string}" "${command_string}" "${api_key_hash}" "${envs_string}")
    config_inputs+=("${model_args[@]}")
    local config_hash
    config_hash=$(printf '%s\0' "${config_inputs[@]}" | sha256sum | awk '{print $1}')

    if container_is_running "${container_name}"; then
        local existing_hash
        existing_hash=$(get_container_config_hash "${container_name}")
        if [[ "${existing_hash}" == "${config_hash}" && -n "${existing_hash}" ]]; then
            echo "✓ Container ${container_name} is already running with matching configuration (hash: ${config_hash})"
            show_container_status "${container_name}"
            echo ""
            echo "To restart the container, stop it first: sudo docker stop ${container_name}"
            exit 0
        fi

        echo "Configuration change detected for ${container_name}. Recreating container..."
        sudo docker stop "${container_name}"
        sudo docker rm "${container_name}" || true
    else
        local existing_hash
        existing_hash=$(get_container_config_hash "${container_name}")
        if [[ -n "${existing_hash}" ]]; then
            echo "Removing stopped container ${container_name} with configuration hash ${existing_hash}..."
            sudo docker rm "${container_name}" || true
        fi
    fi

    echo "Starting vLLM container with provided API key..."
    echo "Image: ${image}"
    echo "Model: ${model}"
    echo "Configuration hash: ${config_hash}"
    if ((${#additional_env_vars[@]} > 0)); then
        echo "Additional environment variables: ${additional_env_vars[*]}"
    fi

    local run_args=(
        -d --runtime nvidia
        --name "${container_name}"
        --label "vllm-config-hash=${config_hash}"
        --label "vllm-model=${model}"
        --label "vllm-host-port=${host_port}"
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
        --env "VLLM_API_KEY=${VLLM_API_KEY}"
        --env "VLLM_LOGGING_LEVEL=${logging_level}"
        -p "${host_port}:8000"
        --ipc=host
        --restart=always
    )
    
    # Add additional VLLM environment variables
    for env_var in "${additional_env_vars[@]}"; do
        run_args+=(--env "${env_var}")
    done

    # Add image
    run_args+=("${image}")

    # Build command arguments: entrypoint_string (if provided) + command_string (if provided) + model_args
    # Docker format: docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
    # If entrypoint_string is provided, it becomes the command, and command_string + model_args become its arguments
    # If entrypoint_string is empty, command_string becomes the command, and model_args become its arguments
    # If both are empty, model_args become the command and its arguments
    
    local cmd_args=()
    
    if [[ -n "${entrypoint_string}" ]]; then
        # Parse entrypoint_string into command and arguments
        read -r -a entrypoint_parts <<< "${entrypoint_string}"
        cmd_args+=("${entrypoint_parts[@]}")
    fi
    
    if [[ -n "${command_string}" ]]; then
        cmd_args+=("${command_string}")
    fi
    
    cmd_args+=("${model_args[@]}")
    
    # Only add command arguments if there are any
    if ((${#cmd_args[@]} > 0)); then
        run_args+=("${cmd_args[@]}")
    fi

    sudo docker run "${run_args[@]}"

    if [[ "${SKIP_DEPLOY_VERIFY:-false}" != "true" ]]; then
        print_header "Step 5: Verifying Deployment"
        if ! verify_vllm_deployment "${host_port}" "${model}" "${VLLM_API_KEY}"; then
            echo "✗ Deployment verification failed. Review container logs for details."
            exit 1
        fi
    else
        echo "Skipping deployment verification because SKIP_DEPLOY_VERIFY=true"
    fi

    print_header "Deployment Complete!"
    show_container_status "${container_name}"
    echo ""
    echo "To view logs, run: sudo docker logs -f ${container_name}"
}

