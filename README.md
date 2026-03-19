# VSR - AI 智能监控与语音转写系统

本项目是一个基于 SenseVoice 的实时语音转写 demo，集成了实时监控、VAD 检测、语音识别及 AI 智能分析功能。

## 主要特性

- **实时监控**：在网页端同步刷新监控画面。
- **高保真 UI**：基于 "The Scholarly Observer" 设计系统，采用玻璃拟态、无边框布局和渐变按钮，提供专业级的操作体验。
- **实时 ASR**：通过 WebSocket 实现极低延迟的语音转写，并在侧边栏动态展示。
- **AI 智能分析**：集成 DeepSeek 智能分析，可在录音结束后一键生成课堂策略建议。
- **系统状态**：实时监控服务端 CPU、内存及模型运行状态。

## 技术栈

- **后端**：Python (FastAPI/Flask), WebSocket API
- **前端**：HTML5, Tailwind CSS, Material Symbols, Marked.js
- **设计原型**：Stitch (VSR 项目)

## 使用方法

1. 启动服务器并访问前端页面。
2. 点击“开始监控”启动录音。
3. 实时转写将出现在“ASR 实时文本流”中。
4. 点击“停止监控”或“获取 AI 建议”触发智能分析报告。
