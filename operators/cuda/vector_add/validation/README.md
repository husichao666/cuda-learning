# 向量加法验证记录

[leetgpu-functional.json](leetgpu-functional.json) 保存第一周五份示例在 LeetGPU Playground 的原始控制台文本，运行日期为 2026-10-28，页面显示 CUDA 11.7、FUNCTIONAL 模式。

| 版本 | 结果 |
| --- | --- |
| 完整示例 | `n=1025: PASS`，退出状态 0 |
| 边界测试 | 10 个长度全部 PASS，退出状态 0 |
| 随机输入 | 同一组 10 个长度全部 PASS，退出状态 0 |
| CUDA events 计时 | 5 组测量完成，正确性 PASS，退出状态 0 |
| 完整调用计时 | 测量完成，正确性 PASS，退出状态 0 |

日志中的时间来自功能执行后端，设备报告 Intel Xeon CPU。它们不用于评估 NVIDIA GPU 性能。后续真实 GPU 实验应另外保存设备信息、命令、输入配置与结果，不覆盖这份历史记录。
