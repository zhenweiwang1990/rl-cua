#!/bin/bash
# shellcheck disable=SC2155

set -e -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR"

# Load environment variables from .env file if it exists
[[ -f ./.env ]] && source ./.env

COMMON_SCRIPT_PATH="${SCRIPT_DIR}/deploy-vllm-common.sh"
if [[ -f "${COMMON_SCRIPT_PATH}" ]]; then
    # shellcheck source=deploy-vllm-common.sh
    source "${COMMON_SCRIPT_PATH}"
else
    echo "Error: Common deployment script not found at ${COMMON_SCRIPT_PATH}"
    exit 1
fi

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

function ssh_docker_ps_json() {
    local ip="$1"
    local ssh_user="${2:-ubuntu}"
    local proxy_opts
    proxy_opts=$(get_proxy_options)

    if [[ -n "${proxy_opts}" ]]; then
        eval command ssh ${proxy_opts} "$ssh_user@${ip}" docker ps --filter label=vllm-model --format json
    else
        command ssh "$ssh_user@${ip}" docker ps --filter label=vllm-model --format json
    fi
}

function get_default_verify_params() {
    local name="$1"
    local model="${VERIFY_MODEL_NAME:-mlfoundations/Gelato-30B-A3B}"
    local port="${HOST_PORT:-8000}"

    echo "${model}|${port}"
}

# Test cases: prompts and image URLs in parallel arrays
declare -a TEST_PROMPTS=(
    "Click the Ollama icon and respond with coordinates in the format X,Y"
    "Click the 'Sync activity' and respond with coordinates in the format X,Y"
    "Click the Use blank canvas and respond with coordinates in the format X,Y"
    "Change Appearance to Auto and respond with coordinates in the format X,Y"
    "Open Microsoft Excel and respond with coordinates in the format X,Y"
    "Share the document and respond with coordinates in the format X,Y"
    "Add an existing Repository from your local drive and respond with coordinates in the format X,Y"
    "Click the 'Send' and respond with coordinates in the format X,Y"
    "download got-oss:20b and respond with coordinates in the format X,Y"
    "Close the 'reconnect with Community' and respond with coordinates in the format X,Y"
)

declare -a TEST_IMAGES=(
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_092343_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_105112_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_113730_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_114530_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_115100_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_130700_gen.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_133750_dev.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_135650_dev.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_140331_dev.png"
    "https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_145207_dev.png"
)

# Global arrays to store test results
declare -a TEST_RESULTS_STATUS=()
declare -a TEST_RESULTS_PROMPT=()
declare -a CONCURRENT_TEST_TIMES=()
declare -a SEQUENTIAL_TEST_TIMES=()

# Function to run a single test case
function run_single_test() {
    local test_index="$1"
    local verify_host="$2"
    local host_port="$3"
    local model_name="$4"
    local api_key="$5"
    local prompt_text="$6"
    local image_url="$7"
    local silent="${8:-false}"  # Optional: if true, suppress output
    local test_mode="${9:-sequential}"  # Optional: "concurrent" or "sequential"
    
    local test_name="Test $((test_index + 1))"
    local start_time
    start_time=$(date +%s.%N)
    
    # Check if this is Step 1 (initial verification test)
    local is_step1=false
    if [[ "${test_index}" == "0" ]] && [[ "${test_mode}" == "sequential" ]] && [[ "${silent}" != "true" ]]; then
        is_step1=true
    fi
    
    if [[ "${silent}" != "true" ]]; then
        printf '\n\033[1;36m%s\033[0m: %s\n' "${test_name}" "${prompt_text}"
        printf 'Image: %s\n' "${image_url}"
        
        # For Step 1, print the curl command in compact format
        if [[ "${is_step1}" == "true" ]]; then
            local endpoint="http://${verify_host}:${host_port}/v1/chat/completions"
            
            # Build compact JSON payload using jq if available, otherwise use printf
            local payload_json
            if command -v jq >/dev/null 2>&1; then
                payload_json=$(jq -nc --arg model "${model_name}" --arg text "${prompt_text}" --arg url "${image_url}" \
                    '{model: $model, messages: [{role: "user", content: [{type: "text", text: $text}, {type: "image_url", image_url: {url: $url}}]}]}')
            else
                # Fallback: simple JSON construction (may fail with special characters)
                payload_json=$(printf '{"model":"%s","messages":[{"role":"user","content":[{"type":"text","text":"%s"},{"type":"image_url","image_url":{"url":"%s"}}]}]}' \
                    "${model_name}" "${prompt_text}" "${image_url}")
            fi
            
            printf '\n\033[1;33mCommand:\033[0m '
            printf 'curl --silent --fail --max-time 60 --header "Content-Type: application/json" --header "Authorization: Bearer %s" --data-raw %s %s\n\n' \
                "${api_key}" "'${payload_json}'" "${endpoint}"
        fi
    fi
    
    local verify_output
    if verify_output=$(
        VERIFY_HOST="${verify_host}" \
        VERIFY_PROMPT_TEXT="${prompt_text}" \
        VERIFY_IMAGE_URL="${image_url}" \
        VERIFY_MAX_WAIT_SECONDS="${VERIFY_MAX_WAIT_SECONDS:-120}" \
        verify_vllm_deployment "${host_port}" "${model_name}" "${api_key}" 2>&1
    ); then
        local end_time
        end_time=$(date +%s.%N)
        local elapsed_time
        elapsed_time=$(awk "BEGIN {printf \"%.2f\", ${end_time} - ${start_time}}")
        
        # Store test result
        TEST_RESULTS_STATUS[$test_index]="PASSED"
        TEST_RESULTS_PROMPT[$test_index]="${prompt_text}"
        
        # Store time based on test mode
        if [[ "${test_mode}" == "concurrent" ]]; then
            CONCURRENT_TEST_TIMES[$test_index]="${elapsed_time}"
        else
            SEQUENTIAL_TEST_TIMES[$test_index]="${elapsed_time}"
        fi
        
        # Format JSON response if jq is available
        local formatted_output
        if command -v jq >/dev/null 2>&1; then
            local json_content
            json_content=$(echo "${verify_output}" | sed -n '/^===========================================$/,/^===========================================$/p' | sed '1d;$d')
            
            if [[ -n "${json_content}" ]]; then
                local formatted_json
                formatted_json=$(echo "${json_content}" | jq -C . 2>/dev/null)
                
                if [[ $? -eq 0 ]] && [[ -n "${formatted_json}" ]]; then
                    local temp_json_file
                    temp_json_file=$(mktemp)
                    echo "${formatted_json}" > "${temp_json_file}"
                    
                    formatted_output=$(echo "${verify_output}" | awk -v json_file="${temp_json_file}" '
                        BEGIN { 
                            in_separator=0
                            separator_count=0
                        }
                        /^===========================================$/ {
                            separator_count++
                            if (separator_count == 1) {
                                print
                                while ((getline line < json_file) > 0) {
                                    print line
                                }
                                close(json_file)
                                in_separator=1
                                next
                            } else if (separator_count == 2) {
                                in_separator=0
                                print
                                next
                            }
                        }
                        in_separator && separator_count == 1 {
                            next
                        }
                        { print }
                    ')
                    rm -f "${temp_json_file}"
                else
                    formatted_output="${verify_output}"
                fi
            else
                formatted_output="${verify_output}"
            fi
        else
            formatted_output="${verify_output}"
        fi
        
        # Extract and display response time
        # For Step 1 (initial verification test), show full response
        # For concurrent/sequential tests (except Step 1), skip response output on success (only show time and status)
        local show_response=true
        if [[ "${test_mode}" == "concurrent" ]]; then
            # Concurrent tests: skip response output for all tests (they run in background)
            show_response=false
        elif [[ "${test_mode}" == "sequential" ]]; then
            # Sequential tests: show response only for Step 1 (test_index 0), skip for others
            if [[ "${test_index}" != "0" ]]; then
                show_response=false
            fi
        fi
        
        if [[ "${silent}" != "true" ]]; then
            local response_time_line
            response_time_line=$(printf '\033[1;36m%s Response time: %s seconds\033[0m' "${test_name}" "${elapsed_time}")
            
            if [[ "${show_response}" == "true" ]]; then
                # Show full response output
                echo "${formatted_output}" | awk -v rt="${response_time_line}" '
                    {
                        if (/Deployment verification succeeded/ && !inserted) {
                            if (prev_line != "") {
                                print prev_line
                            }
                            print rt
                            print ""
                            print $0
                            inserted=1
                        } else if (!inserted) {
                            if (prev_line != "") {
                                print prev_line
                            }
                            prev_line=$0
                        } else {
                            print
                        }
                    }
                    END {
                        if (!inserted && prev_line != "") {
                            print prev_line
                        }
                    }
                '
                printf '\033[1;32m✓ %s PASSED\033[0m\n' "${test_name}"
            else
                # Skip response output, only show success message and time
                printf '\033[1;32m✓ %s PASSED\033[0m\n' "${test_name}"
                printf '%s\n' "${response_time_line}"
            fi
        else
            # In silent mode, only output if show_response is true (for concurrent tests, skip on success)
            if [[ "${show_response}" == "true" ]]; then
                echo "${formatted_output}"
            fi
        fi
        return 0
    else
        local end_time
        end_time=$(date +%s.%N)
        local elapsed_time
        elapsed_time=$(awk "BEGIN {printf \"%.2f\", ${end_time} - ${start_time}}")
        
        # Store test result
        TEST_RESULTS_STATUS[$test_index]="FAILED"
        TEST_RESULTS_PROMPT[$test_index]="${prompt_text}"
        
        # Store time based on test mode
        if [[ "${test_mode}" == "concurrent" ]]; then
            CONCURRENT_TEST_TIMES[$test_index]="${elapsed_time}"
        else
            SEQUENTIAL_TEST_TIMES[$test_index]="${elapsed_time}"
        fi
        
        if [[ "${silent}" != "true" ]]; then
            local verify_error_first_line
            verify_error_first_line=$(printf '%s\n' "${verify_output}" | sed -n '1p')
            local verify_error_remaining
            verify_error_remaining=$(printf '%s\n' "${verify_output}" | sed '1d')
            
            printf '\033[1;31m✗ %s FAILED\033[0m\n' "${test_name}"
            printf '\033[1;31m%s\033[0m\n' "${verify_error_first_line}"
            if [[ -n "${verify_error_remaining}" ]]; then
                printf '%s\n' "${verify_error_remaining}"
            fi
            printf '\033[1;36m%s Response time: %s seconds\033[0m\n' "${test_name}" "${elapsed_time}"
        else
            # In silent mode, still output the error (useful for concurrent tests)
            echo "${verify_output}"
        fi
        return 1
    fi
}

# Function to calculate mean and variance
function calculate_stats() {
    local array_name="$1"
    local count=0
    local sum=0
    local valid_times=()
    
    # Use eval to safely access array by name
    local array_size=0
    if ! eval "array_size=\${#${array_name}[@]}" 2>/dev/null; then
        echo "0.00|0.00"
        return
    fi
    
    if [[ ${array_size} -eq 0 ]]; then
        echo "0.00|0.00"
        return
    fi
    
    # Collect valid times
    for ((i=0; i<array_size; i++)); do
        local time=""
        if ! eval "time=\${${array_name}[$i]}" 2>/dev/null; then
            continue
        fi
        if [[ -n "${time}" ]] && [[ "${time}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            valid_times+=("${time}")
            sum=$(awk "BEGIN {printf \"%.2f\", ${sum} + ${time}}")
            count=$((count + 1))
        fi
    done
    
    if [[ ${count} -eq 0 ]]; then
        echo "0.00|0.00"
        return
    fi
    
    local mean=$(awk "BEGIN {printf \"%.2f\", ${sum} / ${count}}")
    
    # Calculate variance
    local variance_sum=0
    for time in "${valid_times[@]}"; do
        local diff=$(awk "BEGIN {printf \"%.2f\", ${time} - ${mean}}")
        local diff_squared=$(awk "BEGIN {printf \"%.2f\", ${diff} * ${diff}}")
        variance_sum=$(awk "BEGIN {printf \"%.2f\", ${variance_sum} + ${diff_squared}}")
    done
    
    local variance=$(awk "BEGIN {printf \"%.2f\", ${variance_sum} / ${count}}")
    
    echo "${mean}|${variance}"
}

# Function to print test results table
function print_test_results_table() {
    local total_tests=${#TEST_RESULTS_STATUS[@]}
    
    if [[ ${total_tests} -eq 0 ]]; then
        return
    fi
    
    printf '\n\033[1;36m=== Test Results Summary ===\033[0m\n'
    printf '\033[1;97m%-6s %-50s %-15s %-15s\033[0m\n' "Test" "Prompt" "Concurrent(s)" "Sequential(s)"
    printf '\033[1;97m%s\033[0m\n' "-----------------------------------------------------------------------------------------------"
    
    local concurrent_total=0
    local sequential_total=0
    local passed_count=0
    local failed_count=0
    
    for ((i=0; i<total_tests; i++)); do
        local test_num=$((i + 1))
        local prompt="${TEST_RESULTS_PROMPT[$i]}"
        local concurrent_time="${CONCURRENT_TEST_TIMES[$i]:-}"
        local sequential_time="${SEQUENTIAL_TEST_TIMES[$i]:-}"
        
        # Truncate prompt if too long
        if [[ ${#prompt} -gt 48 ]]; then
            prompt="${prompt:0:45}..."
        fi
        
        # Determine status based on time availability
        local concurrent_display="Failed"
        local sequential_display="Failed"
        local concurrent_color=$'\033[1;31m'
        local sequential_color=$'\033[1;31m'
        local concurrent_reset=$'\033[0m'
        local sequential_reset=$'\033[0m'
        
        # Format concurrent time
        if [[ -n "${concurrent_time}" ]] && [[ "${concurrent_time}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            concurrent_total=$(awk "BEGIN {printf \"%.2f\", ${concurrent_total} + ${concurrent_time}}")
            concurrent_display=$(printf "%.2f" "${concurrent_time}")
            concurrent_color=$'\033[1;33m'
            concurrent_reset=$'\033[0m'
        elif [[ -z "${concurrent_time}" ]]; then
            concurrent_display="N/A"
            concurrent_color=$'\033[1;90m'
            concurrent_reset=$'\033[0m'
        fi
        
        # Format sequential time
        if [[ -n "${sequential_time}" ]] && [[ "${sequential_time}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            sequential_total=$(awk "BEGIN {printf \"%.2f\", ${sequential_total} + ${sequential_time}}")
            sequential_display=$(printf "%.2f" "${sequential_time}")
            sequential_color=$'\033[1;33m'
            sequential_reset=$'\033[0m'
            passed_count=$((passed_count + 1))
        elif [[ -z "${sequential_time}" ]]; then
            sequential_display="N/A"
            sequential_color=$'\033[1;90m'
            sequential_reset=$'\033[0m'
            failed_count=$((failed_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
        
        printf '\033[1;36m%-6s\033[0m %-50s %s%-15s%s %s%-15s%s\n' \
            "${test_num}" "${prompt}" \
            "${concurrent_color}" "${concurrent_display}" "${concurrent_reset}" \
            "${sequential_color}" "${sequential_display}" "${sequential_reset}"
    done
    
    printf '\033[1;97m%s\033[0m\n' "-----------------------------------------------------------------------------------------------"
    
    # Calculate statistics
    local concurrent_stats
    concurrent_stats=$(calculate_stats CONCURRENT_TEST_TIMES 2>/dev/null || echo "0.00|0.00")
    IFS='|' read -r concurrent_mean concurrent_variance <<< "${concurrent_stats}"
    
    local sequential_stats
    sequential_stats=$(calculate_stats SEQUENTIAL_TEST_TIMES 2>/dev/null || echo "0.00|0.00")
    IFS='|' read -r sequential_mean sequential_variance <<< "${sequential_stats}"
    
    # Print totals
    local concurrent_total_display=""
    local sequential_total_display=""
    if (( $(awk "BEGIN {print (${concurrent_total} > 0)}") )); then
        concurrent_total_display=$(printf "%.2f" "${concurrent_total}")
    else
        concurrent_total_display="N/A"
    fi
    
    if (( $(awk "BEGIN {print (${sequential_total} > 0)}") )); then
        sequential_total_display=$(printf "%.2f" "${sequential_total}")
    else
        sequential_total_display="N/A"
    fi
    
    printf '\033[1;97m%-6s %-50s \033[1;33m%-15s\033[0m \033[1;33m%-15s\033[0m\n' \
        "Total" "${total_tests} tests" "${concurrent_total_display}" "${sequential_total_display}"
    
    if [[ ${failed_count} -gt 0 ]]; then
        printf '\033[1;31mFailed/Timeout: %d\033[0m\n' "${failed_count}"
    fi
    if [[ ${passed_count} -gt 0 ]]; then
        printf '\033[1;32mPassed: %d\033[0m\n' "${passed_count}"
    fi
    
    # Print statistics
    printf '\n\033[1;36m=== Statistics ===\033[0m\n'
    printf '\033[1;97m%-20s %-15s %-15s\033[0m\n' "Mode" "Mean (s)" "Variance"
    printf '\033[1;97m%s\033[0m\n' "--------------------------------------------------------"
    
    if [[ "${concurrent_mean}" != "0.00" ]] || [[ ${#CONCURRENT_TEST_TIMES[@]} -gt 0 ]]; then
        printf '\033[1;97m%-20s\033[0m \033[1;33m%-15s\033[0m \033[1;33m%-15s\033[0m\n' \
            "Concurrent" "${concurrent_mean}" "${concurrent_variance}"
    fi
    
    if [[ "${sequential_mean}" != "0.00" ]] || [[ ${#SEQUENTIAL_TEST_TIMES[@]} -gt 0 ]]; then
        printf '\033[1;97m%-20s\033[0m \033[1;33m%-15s\033[0m \033[1;33m%-15s\033[0m\n' \
            "Sequential" "${sequential_mean}" "${sequential_variance}"
    fi
}

function prompt_verify_vllm_service() {
    local default_ip="$1"
    local ssh_user="${2:-ubuntu}"
    local fallback_model="$3"
    local fallback_port="$4"

    if ! declare -F verify_vllm_deployment >/dev/null 2>&1; then
        echo "Error: verify_vllm_deployment is unavailable (deploy-vllm-common.sh not sourced)."
        exit 1
    fi

    local verify_host
    verify_host="${default_ip}"

    local host_port
    host_port="${fallback_port:-${HOST_PORT:-8000}}"

    local model_name
    model_name="${VERIFY_MODEL_NAME:-${fallback_model:-mlfoundations/Gelato-30B-A3B}}"

    local container_status_output=""
    container_status_output=$(ssh_docker_ps_json "${verify_host}" "${ssh_user}" 2>/dev/null || true)
    local container_json_entries=()
    if [[ -n "${container_status_output}" ]]; then
        while IFS= read -r line; do
            [[ -z "${line}" ]] && continue
            if [[ "${line}" == \{* ]] && [[ "${line}" == *\} ]]; then
                local parsed_line
                parsed_line=$(printf '%s\n' "${line}" | jq -r '
                  (.Labels // "") as $labels
                  | ($labels
                      | split(",")
                      | map(select(length>0) | split("=") | {(.[0]): (.[1] // "")})
                      | add) as $L
                  | "\(.Names // "")|\($L["vllm-model"] // "")|\($L["vllm-host-port"] // "")"
                ' 2>/dev/null)
                if [[ -n "${parsed_line}" && "${parsed_line}" != "null" ]]; then
                    container_json_entries+=("${parsed_line}")
                fi
            fi
        done <<< "$(printf '%s\n' "${container_status_output}" | sed '/^[[:space:]]*$/d')"
    fi

    local selected_entry=""
    if [[ ${#container_json_entries[@]} -gt 0 ]]; then
        local derived_list=("${container_json_entries[@]}")
        if [[ ${#derived_list[@]} -gt 1 ]]; then
            local prioritized_entries=()
            local deprioritized_entries=()
            for entry in "${derived_list[@]}"; do
                if [[ "${entry}" == *"${model_name}"* ]]; then
                    prioritized_entries+=("${entry}")
                else
                    deprioritized_entries+=("${entry}")
                fi
            done
            derived_list=("${prioritized_entries[@]}" "${deprioritized_entries[@]}")
            printf "\033[1;36m%-4s\033[0m \033[1;97m%-30s\033[0m \033[1;35m%-45s\033[0m \033[1;33m%-8s\033[0m\n" "No." "Container" "Model" "Port"
            local idx=1
            for entry in "${derived_list[@]}"; do
                local name parsed_model parsed_port
                IFS='|' read -r name parsed_model parsed_port <<< "${entry}"
                name=${name:-"<unknown>"}
                parsed_model=${parsed_model:-"<unknown>"}
                parsed_port=${parsed_port:-"<unset>"}
                printf "\033[1;36m%-4s\033[0m \033[1;97m%-30s\033[0m \033[1;35m%-45s\033[0m \033[1;33m%-8s\033[0m\n" "${idx}." "${name}" "${parsed_model}" "${parsed_port}"
                idx=$((idx + 1))
            done
            container_json_entries=("${derived_list[@]}")
            local selection
            read -p "Select container to verify (1-${#container_json_entries[@]}, default 1): " selection
            selection=${selection:-1}
            if ! [[ "${selection}" =~ ^[0-9]+$ ]] || (( selection < 1 || selection > ${#container_json_entries[@]} )); then
                selection=1
            fi
            selected_entry="${container_json_entries[$((selection-1))]}"
        else
            selected_entry="${derived_list[0]}"
        fi

        if [[ -n "${selected_entry}" ]]; then
            local selected_container selected_model selected_port
            IFS='|' read -r selected_container selected_model selected_port <<< "${selected_entry}"
            selected_container=${selected_container:-"unknown-container"}
            if [[ -n "${selected_model}" ]]; then
                model_name="${selected_model}"
            fi
            if [[ -n "${selected_port}" ]]; then
                host_port="${selected_port}"
            fi
            printf '\nVerifying container %s (model: %s, port: %s)\n\n' "${selected_container}" "${model_name}" "${host_port}"
        fi
    fi

    host_port=${host_port:-${HOST_PORT:-8000}}
    model_name=${model_name:-${VERIFY_MODEL_NAME:-mlfoundations/Gelato-30B-A3B}}

    local api_key
    api_key="${VLLM_API_KEY:-}"
    if [[ -z "${api_key}" ]]; then
        read -s -p "Enter VLLM API key: " api_key
        echo ""
        if [[ -z "${api_key}" ]]; then
            echo "Error: VLLM API key is required for verification."
            exit 1
        fi
    fi

    # Use test cases from array, or fallback to environment variables
    local use_test_cases=true
    if [[ -n "${VERIFY_PROMPT_TEXT:-}" ]] && [[ -n "${VERIFY_IMAGE_URL:-}" ]]; then
        # If environment variables are set, use single test mode (backward compatibility)
        use_test_cases=false
    fi

    if [[ "${use_test_cases}" == "true" ]] && [[ ${#TEST_PROMPTS[@]} -gt 0 ]] && [[ ${#TEST_IMAGES[@]} -gt 0 ]]; then
        # Ensure arrays have the same length
        if [[ ${#TEST_PROMPTS[@]} -ne ${#TEST_IMAGES[@]} ]]; then
            echo "Error: TEST_PROMPTS and TEST_IMAGES arrays must have the same length"
            return 1
        fi
        
        # Multi-test mode: single test first, then sequential tests, then concurrent tests
        printf '\n\033[1;33m=== Running Multi-Test Verification ===\033[0m\n'
        printf 'Total test cases: %d\n\n' "${#TEST_PROMPTS[@]}"
        
        # Step 1: Run first test case (single test)
        printf '\033[1;36m=== Step 1: Initial Verification Test ===\033[0m\n'
        local first_prompt="${TEST_PROMPTS[0]}"
        local first_image="${TEST_IMAGES[0]}"
        
        if ! run_single_test 0 "${verify_host}" "${host_port}" "${model_name}" "${api_key}" "${first_prompt}" "${first_image}" "false" "sequential"; then
            printf '\n\033[1;31mInitial test failed. Skipping sequential and concurrent tests.\033[0m\n'
            return 1
        fi
        
        # Step 2: Run all tests sequentially
        printf '\n\033[1;36m=== Step 2: Sequential Verification Tests ===\033[0m\n'
        printf 'Running all %d tests sequentially...\n\n' "${#TEST_PROMPTS[@]}"
        
        local seq_failed_count=0
        local seq_success_count=0
        
        for ((i=0; i<${#TEST_PROMPTS[@]}; i++)); do
            local test_prompt="${TEST_PROMPTS[$i]}"
            local test_image="${TEST_IMAGES[$i]}"
            
            if run_single_test "${i}" "${verify_host}" "${host_port}" "${model_name}" "${api_key}" "${test_prompt}" "${test_image}" "false" "sequential"; then
                seq_success_count=$((seq_success_count + 1))
            else
                seq_failed_count=$((seq_failed_count + 1))
            fi
        done
        
        # Step 3: Run all tests concurrently (including the first one)
        if [[ ${#TEST_PROMPTS[@]} -gt 0 ]]; then
            printf '\n\033[1;36m=== Step 3: Concurrent Verification Tests ===\033[0m\n'
            printf 'Running all %d tests concurrently...\n\n' "${#TEST_PROMPTS[@]}"
            
            local pids=()
            local temp_dir
            temp_dir=$(mktemp -d)
            
            # Launch all tests concurrently (including the first one)
            for ((i=0; i<${#TEST_PROMPTS[@]}; i++)); do
                local test_prompt="${TEST_PROMPTS[$i]}"
                local test_image="${TEST_IMAGES[$i]}"
                
                (
                    local test_output_file="${temp_dir}/test_${i}.out"
                    local test_time_file="${temp_dir}/test_${i}.time"
                    local test_status_file="${temp_dir}/test_${i}.status"
                    
                    # Capture time in a way that can be read by parent process
                    local start_time
                    start_time=$(date +%s.%N)
                    
                    # Run test with output captured (silent=true to suppress output, redirect to file)
                    if run_single_test "${i}" "${verify_host}" "${host_port}" "${model_name}" "${api_key}" "${test_prompt}" "${test_image}" "true" "concurrent" > "${test_output_file}" 2>&1; then
                        echo "SUCCESS" > "${test_status_file}"
                    else
                        echo "FAILED" > "${test_status_file}"
                    fi
                    
                    local end_time
                    end_time=$(date +%s.%N)
                    local elapsed_time
                    elapsed_time=$(awk "BEGIN {printf \"%.2f\", ${end_time} - ${start_time}}")
                    echo "${elapsed_time}" > "${test_time_file}"
                ) &
                pids+=($!)
            done
            
            # Wait for all tests to complete and collect results
            local failed_count=0
            local success_count=0
            
            for ((i=0; i<${#pids[@]}; i++)); do
                wait "${pids[$i]}"
                local array_index="${i}"
                local status_file="${temp_dir}/test_${array_index}.status"
                local output_file="${temp_dir}/test_${array_index}.out"
                local time_file="${temp_dir}/test_${array_index}.time"
                
                # Read time from file and store in array
                if [[ -f "${time_file}" ]]; then
                    local elapsed_time
                    elapsed_time=$(cat "${time_file}")
                    if [[ -n "${elapsed_time}" ]] && [[ "${elapsed_time}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                        CONCURRENT_TEST_TIMES[$array_index]="${elapsed_time}"
                    fi
                fi
                
                if [[ -f "${status_file}" ]] && [[ "$(cat "${status_file}")" == "SUCCESS" ]]; then
                    success_count=$((success_count + 1))
                else
                    failed_count=$((failed_count + 1))
                fi
                
                # Display test output with separator (only show output for failed tests)
                if [[ -f "${output_file}" ]] && [[ -f "${status_file}" ]] && [[ "$(cat "${status_file}")" != "SUCCESS" ]]; then
                    printf '\n\033[1;33m--- Test %d Output ---\033[0m\n' "$((array_index + 1))"
                    # Only show output if file is not empty
                    if [[ -s "${output_file}" ]]; then
                        cat "${output_file}"
                    else
                        printf '\033[1;90m(No output captured)\033[0m\n'
                    fi
                fi
            done
            
            # Cleanup
            rm -rf "${temp_dir}"
            
            # Print results table
            print_test_results_table
            
            # Summary
            printf '\n\033[1;36m=== Final Summary ===\033[0m\n'
            printf '\033[1;32mSequential tests - Passed: %d\033[0m\n' "${seq_success_count}"
            if [[ ${seq_failed_count} -gt 0 ]]; then
                printf '\033[1;31mSequential tests - Failed: %d\033[0m\n' "${seq_failed_count}"
            fi
            printf '\033[1;32mConcurrent tests - Passed: %d\033[0m\n' "${success_count}"
            if [[ ${failed_count} -gt 0 ]]; then
                printf '\033[1;31mConcurrent tests - Failed: %d\033[0m\n' "${failed_count}"
            fi
            
            # Return error if any test failed
            if [[ ${seq_failed_count} -gt 0 ]] || [[ ${failed_count} -gt 0 ]]; then
                return 1
            else
                printf '\033[1;32mAll tests passed!\033[0m\n'
                return 0
            fi
        else
            # Single test case scenario (should not happen with multi-test mode)
            printf '\n\033[1;32mSequential test passed!\033[0m\n'
            print_test_results_table
            return 0
        fi
    else
        # Single test mode (backward compatibility)
        local prompt_text
        prompt_text="${VERIFY_PROMPT_TEXT:-Click the Ollama icon and respond with coordinates in the format X,Y}"

        local image_url
        image_url="${VERIFY_IMAGE_URL:-https://gbox-cua-datasets.s3.amazonaws.com/eval-gym/gbox-bench/v1/mac/251003_092343_gen.png}"

        local payload_compact
        payload_compact="$(jq -cn \
            --arg model "${model_name}" \
            --arg prompt "${prompt_text}" \
            --arg image "${image_url}" \
            '{
                model: $model,
                messages: [
                    {
                        role: "user",
                        content: [
                            {type: "text", text: $prompt},
                            {type: "image_url", image_url: {url: $image}}
                        ]
                    }
                ]
            }'
        )"

        local curl_command
        curl_command=$(
            cat <<EOF
curl \\
  -X POST \\
  "http://${verify_host}:${host_port}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${api_key}" \\
  --data '${payload_compact}'
EOF
        )

        printf 'Verification command:\n'
        echo "==========================================="
        printf '\033[1;32m%s\033[0m;\n' "${curl_command}"
        echo "==========================================="
        echo ""

        local start_time
        start_time=$(date +%s.%N)
        local verify_output
        if verify_output=$(
            VERIFY_HOST="${verify_host}" \
            VERIFY_PROMPT_TEXT="${prompt_text}" \
            VERIFY_IMAGE_URL="${image_url}" \
            VERIFY_MAX_WAIT_SECONDS="${VERIFY_MAX_WAIT_SECONDS:-120}" \
            verify_vllm_deployment "${host_port}" "${model_name}" "${api_key}"
        ); then
            local end_time
            end_time=$(date +%s.%N)
            local elapsed_time
            elapsed_time=$(awk "BEGIN {printf \"%.2f\", ${end_time} - ${start_time}}")
            local response_time_line
            response_time_line=$(printf '\033[1;36mResponse time: %s seconds\033[0m' "${elapsed_time}")
            
            # Format JSON response if jq is available and response is valid JSON
            local formatted_output
            if command -v jq >/dev/null 2>&1; then
                # Extract JSON content between separator lines
                local json_content
                json_content=$(echo "${verify_output}" | sed -n '/^===========================================$/,/^===========================================$/p' | sed '1d;$d')
                
                if [[ -n "${json_content}" ]]; then
                    # Try to format JSON with jq (with color output enabled)
                    local formatted_json
                    formatted_json=$(echo "${json_content}" | jq -C . 2>/dev/null)
                    
                    if [[ $? -eq 0 ]] && [[ -n "${formatted_json}" ]]; then
                        # Use temporary file to handle multi-line JSON
                        local temp_json_file
                        temp_json_file=$(mktemp)
                        echo "${formatted_json}" > "${temp_json_file}"
                        
                        # Replace JSON section with formatted JSON
                        formatted_output=$(echo "${verify_output}" | awk -v json_file="${temp_json_file}" '
                            BEGIN { 
                                in_separator=0
                                separator_count=0
                            }
                            /^===========================================$/ {
                                separator_count++
                                if (separator_count == 1) {
                                    print
                                    # Print formatted JSON from file
                                    while ((getline line < json_file) > 0) {
                                        print line
                                    }
                                    close(json_file)
                                    in_separator=1
                                    next
                                } else if (separator_count == 2) {
                                    in_separator=0
                                    print
                                    next
                                }
                            }
                            in_separator && separator_count == 1 {
                                # Skip original JSON lines
                                next
                            }
                            { print }
                        ')
                        rm -f "${temp_json_file}"
                    else
                        formatted_output="${verify_output}"
                    fi
                else
                    formatted_output="${verify_output}"
                fi
            else
                formatted_output="${verify_output}"
            fi
            
            # Insert response time before "✓ Deployment verification succeeded"
            # Remove empty line before success message, add response time, then add empty line after
            echo "${formatted_output}" | awk -v rt="${response_time_line}" '
                {
                    if (/Deployment verification succeeded/ && !inserted) {
                        # If previous line was empty, skip it; otherwise print it
                        if (prev_line != "") {
                            print prev_line
                        }
                        print rt
                        print ""
                        print $0
                        inserted=1
                    } else if (!inserted) {
                        if (prev_line != "") {
                            print prev_line
                        }
                        prev_line=$0
                    } else {
                        print
                    }
                }
                END {
                    if (!inserted && prev_line != "") {
                        print prev_line
                    }
                }
            '
        else
            local end_time
            end_time=$(date +%s.%N)
            local elapsed_time
            elapsed_time=$(awk "BEGIN {printf \"%.2f\", ${end_time} - ${start_time}}")
            local verify_error_first_line
            verify_error_first_line=$(printf '%s\n' "${verify_output}" | sed -n '1p')
            local verify_error_remaining
            verify_error_remaining=$(printf '%s\n' "${verify_output}" | sed '1d')
            printf '\033[1;31m%s\033[0m\n' "${verify_error_first_line}"
            if [[ -n "${verify_error_remaining}" ]]; then
                printf '%s\n' "${verify_error_remaining}"
            fi
            printf '\n\033[1;36mResponse time: %s seconds\033[0m\n' "${elapsed_time}"
            return 1
        fi
    fi
}

# Main execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 <instance_ip> [ssh_user] [fallback_model] [fallback_port]"
        exit 1
    fi
    
    default_ip="$1"
    ssh_user="${2:-ubuntu}"
    fallback_model="${3:-}"
    fallback_port="${4:-}"
    
    # If fallback_model and fallback_port are not provided, try to get defaults
    if [[ -z "${fallback_model}" ]] && [[ -z "${fallback_port}" ]]; then
        default_params=$(get_default_verify_params "")
        IFS='|' read -r fallback_model fallback_port <<< "${default_params}"
    fi
    
    prompt_verify_vllm_service "${default_ip}" "${ssh_user}" "${fallback_model}" "${fallback_port}"
fi

