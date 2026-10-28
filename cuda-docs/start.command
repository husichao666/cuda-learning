#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if ! command -v node >/dev/null 2>&1; then
  echo '请先安装 Node.js，推荐版本 24：https://nodejs.org/'
  read -r '?按回车退出'
  exit 1
fi
if [[ ! -f node_modules/vitepress/bin/vitepress.js ]]; then
  echo '请先在此目录运行 pnpm install --frozen-lockfile 安装依赖。'
  read -r '?按回车退出'
  exit 1
fi
echo '启动 CUDA 文档站；终端中按 Control+C 可停止。'
exec node node_modules/vitepress/bin/vitepress.js dev docs --host 127.0.0.1 --port 5173
