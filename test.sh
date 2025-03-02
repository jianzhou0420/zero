#!/bin/bash
# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <window_match_string> <program_command> [args...]"
    exit 1
fi

# 第一个参数作为窗口匹配字符串，后面的参数作为要执行的程序及其参数
window_match=$1
shift
program_cmd="$@"

# 检查 wmctrl 是否安装
if ! command -v wmctrl &> /dev/null; then
    echo "wmctrl 未安装，请先安装：sudo apt-get install wmctrl"
    exit 1
fi

# 检查 xrandr 是否安装
if ! command -v xrandr &> /dev/null; then
    echo "xrandr 未安装，请先安装：sudo apt-get install xrandr"
    exit 1
fi

# 获取第三个连接显示器的信息（按 xrandr 输出的顺序）
third_monitor_line=$(xrandr | grep " connected" | sed -n '3p')
if [ -z "$third_monitor_line" ]; then
    echo "无法获取第三个显示器信息，请检查连接状态。"
    exit 1
fi

# 提取第三个显示器的几何信息，例如：1920x1080+3840+0
geometry=$(echo "$third_monitor_line" | grep -o "[0-9]\+x[0-9]\++[0-9]\++[0-9]\+")
if [ -z "$geometry" ]; then
    echo "无法解析第三个显示器的几何信息。"
    exit 1
fi

# 解析几何字符串：宽x高+x_offset+y_offset
IFS='x+' read -r monitor_width monitor_height xpos ypos <<< "$geometry"
echo "第三个显示器分辨率：${monitor_width}x${monitor_height}，起始坐标：(${xpos},${ypos})"

# 启动指定的程序
$program_cmd &

# 等待窗口启动，根据匹配字符串判断窗口是否出现
echo "等待窗口 \"$window_match\" 出现..."
while ! wmctrl -l | grep -q "$window_match"; do
    sleep 0.5
done

# 移动窗口到第三个显示器的左上角，保持原有大小（-1 表示不修改宽度和高度）
wmctrl -r "$window_match" -e 0,"$xpos","$ypos",-1,-1

echo "程序窗口已移动到第三个显示器的左上角。"
