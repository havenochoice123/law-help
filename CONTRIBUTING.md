# Contributing

本仓库采用轻量化协作流程（适合 3 周冲刺）。

## 分支规范
- 统一从 `main` 拉取个人 feature 分支。
- 分支命名建议：`feat/<topic>`、`fix/<topic>`、`docs/<topic>`。

示例：
- feat/train-lora
- feat/index
- docs/readme

## 提交规范
- 提交格式：`type: message`
- 常用类型：`feat` / `fix` / `docs` / `chore`
- 类型说明：
  - `feat`: 新功能或新增能力
  - `fix`: 缺陷修复或错误纠正
  - `docs`: 文档更新或说明补充
  - `chore`: 构建/依赖/脚手架等杂项维护

示例：
- feat: add lora quick training script
- fix: handle empty chunks
- docs: update README

## Pull Request 规范
- 每个 PR 需要至少 1 人 review（检查逻辑、可运行性与命名一致性）。
- PR 描述需包含：变更说明、运行方式、验证结果（如有）。

## 协作流程（推荐）
1) 更新 main
```powershell
git checkout main
git pull
```

2) 创建 feature 分支
```powershell
git checkout -b feat/your-topic
```

3) 开发与提交
```powershell
git add .
git commit -m "feat: your change"
```

4) 推送并创建 PR
```powershell
git push -u origin feat/your-topic
```


## 冲突处理（快速）
```powershell
git checkout main
git pull
git checkout feat/your-topic
git merge main
# 解决冲突后

git add <conflict-files>
git commit -m "fix: resolve merge conflicts"
```
