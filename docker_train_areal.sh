#!/bin/bash
# AReaL 训练 Docker 启动脚本

set -e

# ============ 解析命令行参数 ============
CONFIG_FILE="configs/cua_grpo.yaml"  # 默认配置文件
COMPOSE_FILE="docker-compose.areal.yml"  # 默认 compose 文件
PROJECT_NAME="cua-areal"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --compose|-f)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE     Training config file (default: configs/cua_grpo.yaml)"
            echo "  -f, --compose FILE    Docker Compose file (default: docker-compose.areal.yml)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default config"
            echo "  $0 --config configs/cua_grpo_single_gpu.yaml"
            echo "  $0 -c configs/cua_grpo_single_gpu.yaml -f docker-compose.areal.single-gpu.yml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============ 配置 ============
PROJECT_NAME="cua-areal"

# ============ 加载环境变量 ============
if [ -f .env ]; then
    # 安全地加载 .env 文件，只导出格式正确的 KEY=VALUE 行
    # 忽略注释行（以 # 开头）和空行
    # 处理行内注释（KEY=VALUE # comment）
    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过空行和注释行
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        # 移除行内注释
        line="${line%%#*}"
        # 去除首尾空格
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        
        # 只导出格式正确的 KEY=VALUE
        if [[ "$line" =~ ^[[:alnum:]_]+= ]]; then
            export "$line"
        fi
    done < .env
fi

# ============ 检查配置文件 ============
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please specify a valid config file with --config option"
    exit 1
fi

# ============ 检查必要的环境变量 ============
if [ -z "$GBOX_API_KEY" ]; then
    echo "Error: GBOX_API_KEY is not set"
    echo "Please set it in .env file or as environment variable"
    exit 1
fi

# ============ 显示配置信息 ============
echo "=========================================="
echo "AReaL Training Configuration"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Compose file: $COMPOSE_FILE"
echo "=========================================="
echo ""

# ============ 构建镜像 ============
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" build

# ============ 启动服务 ============
echo "Starting services..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d vllm

# 等待 vLLM 启动
echo "Waiting for vLLM to be ready..."
while ! docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" exec -T vllm curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "  Waiting..."
    sleep 10
done
echo "vLLM is ready!"

# ============ 启动训练 ============
echo "Starting AReaL training..."
echo "Using config: $CONFIG_FILE"
# 通过环境变量传递配置文件路径给 docker-compose
export CONFIG_FILE="$CONFIG_FILE"
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up trainer

# ============ 清理 ============
echo "Training complete. Cleaning up..."
docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down

echo "Done!"

