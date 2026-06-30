# CI/CD 说明

本文档说明本仓库的 CI/CD 是怎么组织的、GitHub Actions 在这个项目里到底做了什么、
GitHub Pages 在本项目中的发布流程是什么、以及这套文档流水线如何从修改源码一路走到上线。

## 1. 本仓库里的 CI/CD 是什么

本仓库当前的 CI/CD 重点放在**文档交付**，而不是模型训练自动化。

- **CI**：自动检查并构建文档源码
- **CD**：自动把生成好的静态网页发布到 GitHub Pages

当前流程非常清晰：

1. 代码推送到 `main`
2. GitHub Actions 启动文档 workflow
3. Sphinx 从 `doc/source` 构建 HTML
4. 构建结果作为 Pages artifact 上传
5. GitHub Pages 发布站点

## 2. GitHub Actions 是什么

GitHub Actions 是 GitHub 自带的托管式 CI/CD 系统。

从职责上看，它和 Jenkins 属于同一类东西，都会负责：

- 监听仓库事件，例如 push、手动触发
- 拉取代码
- 运行构建脚本
- 运行测试或校验
- 发布产物

差别主要在运维方式：

- GitHub Actions 直接集成在 GitHub 里
- Jenkins 通常是自己部署、自己维护

## 3. 本项目怎么使用 GitHub Actions

本仓库当前使用的工作流文件是：

- `.github/workflows/docs.yml`

它专门负责文档发布。

### 触发条件

这个 workflow 会在以下情况运行：

- 有代码 push 到 `main`
- 在 GitHub Actions 页面手动触发

### 权限

当前 workflow 申请了这些权限：

- `contents: read`
- `pages: write`
- `id-token: write`

这些权限是通过 GitHub Actions 部署 Pages 所需要的。

### 并发控制

这个 workflow 还配置了：

- Pages 专用并发组
- `cancel-in-progress: true`

意思是：如果很短时间内连续触发多个文档部署，旧的正在运行中的部署可以被新的替代掉。
这本质上就是一种常见的 CI/CD 优化策略。

## 4. 本项目文档发布链路

当前文档路径是：

- 源码目录：`doc/source/`
- 本地构建结果：`doc/build/html/`
- 线上发布目标：GitHub Pages

### 本地构建

本地构建命令是：

```bash
make -C doc html
```

它会调用 Sphinx，在下面这个目录生成静态网页：

```text
doc/build/html/
```

### 为什么 `doc/build/` 不跟踪

`doc/build/` 是生成产物，不提交到 Git 是合理设计。

本项目的思路是：

- 用 Git 跟踪文档源码
- 用 GitHub Actions 构建并发布产物

这样可以避免：

- 大量生成 HTML 文件污染仓库
- 源码和编译结果不同步
- 每次改文档都同时提交两套内容

## 5. GitHub Pages 在本项目里怎么工作

本项目使用的是：

- **GitHub Pages**
- **Source = GitHub Actions**

这意味着它**不是**从一个已提交的 `docs/` 目录直接发布，也不是靠仓库里长期保留
静态网页文件来发布。

而是：

1. Actions 构建 HTML
2. Actions 上传构建产物
3. Pages 从这个产物完成发布

## 6. 本项目实际是怎么启用 Pages 的

这次启用本仓库 Pages 的真实流程就是：

1. 打开 GitHub 仓库页面
2. 进入 `Settings`
3. 打开 `Pages`
4. 找到 `Build and deployment`
5. 把 `Source` 设置成 `GitHub Actions`

完成这一步以后，这个仓库才真正允许 Actions 创建 Pages deployment。

## 7. 第一次为什么部署失败

第一次部署失败时，并不是 workflow 文件本身有问题，而是：

- `build` job 成功了
- artifact 也上传成功了
- 但 GitHub Pages 还没有真正启用完成

所以失败点出现在：

- Pages deployment creation failed

也就是说，GitHub 还不允许这个仓库创建 Pages 发布记录。

在 `Settings -> Pages -> Source -> GitHub Actions` 配好以后，正确修复流程就是：

1. 回到失败的 workflow run
2. 点击 `Re-run all jobs`

这样同一个 workflow 就能重新跑通并完成发布。

## 8. 当前文档 workflow 的结构

当前 workflow 里有两个 job：

### `build`

这个 job 负责：

- checkout 仓库
- 安装 Python
- 安装文档依赖
- 运行 `make -C doc html`
- 上传 `doc/build/html` 作为 Pages artifact

### `deploy`

这个 job 负责：

- 等待 `build`
- 接收前面上传的 artifact
- 把它部署到 GitHub Pages

## 9. 以后这个流程怎么使用

### 正常更新流程

以后如果改文档：

1. 修改 `doc/source/` 下的文档源码
2. 本地可选执行：
   ```bash
   make -C doc html
   ```
3. commit 并 push 到 `main`
4. GitHub Actions 自动运行
5. GitHub Pages 自动更新线上站点

### 如果部署失败

建议按这个顺序排查：

1. 打开 `Actions`
2. 看是 `build` 失败还是 `deploy` 失败
3. 打开 `Settings -> Pages`
4. 确认 `Source` 还是 `GitHub Actions`
5. 如果是配置改完后才生效，就重新运行 workflow

## 10. 免费用户的 GitHub Actions 有什么限制

限制和仓库类型关系很大。

### Public repository

对于公开仓库，GitHub Actions 一般会宽松得多。

### Private repository

对于私有仓库，免费额度通常会更容易受限，主要体现在：

- 总 runner 分钟数
- artifact / log 保留
- 并发能力

另外还要注意：

- Linux runner 最省
- macOS runner 最贵

本仓库当前使用 Linux 路线，这是控制成本的正确默认选择。

## 11. 高频 push 场景下怎么办

在更大规模的工程团队里，不会让每次 push 都完整跑到底。

常见做法包括：

- 自动取消旧的运行
- 只保留最新提交的运行
- 按路径触发不同 workflow
- 按分支拆轻量检查和重量部署
- 使用缓存
- 用 self-hosted runner 扩容

本仓库虽然不复杂，但已经用了其中一个很重要的思想：

- Pages 部署并发控制

## 12. 为什么这套设计适合本项目

这个仓库是研究 / 代码 / 文档项目，不是一个高频后端服务系统。

所以当前方案合适的原因是：

- 简单
- 可维护
- 运维负担小
- 和 GitHub 深度集成
- 足够支持文档发布

## 13. 为什么用了 CI/CD 之后网页也不是瞬间刷新

用了 CI/CD，并不等于你一 `push`，全网访问到的网页内容就会立刻同步完成。

真实链路其实是：

1. `push` 到 GitHub
2. GitHub Actions 启动 workflow
3. workflow 构建站点
4. 上传 Pages artifact
5. GitHub Pages 创建 deployment
6. 新版本再通过 GitHub 的发布链路逐步生效

所以，CI/CD 主要解决的是：

- 自动化
- 可追踪
- 可复现

它并不保证“浏览器下一秒就一定看到新页面”。

现实里，页面生效时间常常是几十秒到几分钟；有些仓库更慢，甚至可能更久。这里的瓶颈
未必是浏览器缓存，也可能就是 GitHub Pages 自己的部署和生效链路。

这也是为什么一个纯静态的 HTML/CSS/JS 个人网站，即便：

- 浏览器禁用了缓存
- CSS/JS 做了版本号

更新之后依然可能不是立刻可见。因为问题不一定出在前端资源缓存，而可能出在：

- Pages 还没完成部署
- GitHub 侧发布还没完全生效
- 访问请求还没稳定落到新版本

因此更准确的区分应该是：

- **CI/CD** 负责回答：新版本有没有被正确构建并正确发布
- **Pages 刷新延迟** 负责回答：新版本多快能在外部访问中稳定可见

这两件事相关，但不是一回事。

## 14. 如果将来需要，也可以继续扩展

- 测试 workflow
- lint workflow
- 打包 workflow
- benchmark 或复现实验检查

## 15. 当前文档公开地址

本仓库预期的 Pages 地址是：

```text
https://billzi2016.github.io/DeepChrInteract-v2/
```

## 16. 关键文件

- workflow：`.github/workflows/docs.yml`
- Sphinx 源码根目录：`doc/source/`
- 本地构建目录：`doc/build/html/`
- 英文仓库说明：`README.md`
- 中文仓库说明：`README_CN.md`
