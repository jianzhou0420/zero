#!/bin/bash

# 打开 terminator 并执行命令，后台运行
terminator -x bash -c "echo Hello from positioned Terminator; exec bash" &

# 等待窗口出现（你也可以 sleep 1 ）
sleep 1

# 获取窗口 ID（匹配 terminator 名字）
win_id=$(xdotool search --onlyvisible --class terminator | tail -1)

# 将窗口移动到特定位置（例如 100,200）
xdotool windowmove "$win_id" 100 200
