import { defineConfig } from 'vitepress'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import footnote from 'markdown-it-footnote'

const articles = JSON.parse(readFileSync(new URL('../articles/sidebar.json', import.meta.url), 'utf8'))

export default defineConfig({
  lang: 'zh-CN',
  title: 'CUDA 学习笔记',
  description: '从 GPU 硬件与第一个 CUDA 程序开始，记录算子学习、代码验证与实验。',
  cleanUrls: true,
  outDir: fileURLToPath(new URL('../../../build/pages', import.meta.url)),
  markdown: { config: (md) => md.use(footnote) },
  themeConfig: {
    siteTitle: 'CUDA 学习笔记',
    nav: [
      { text: '笔记目录', link: '/articles/' },
      { text: 'CUDA 中文指南', link: 'https://husichao666.github.io/cuda-learning/guide/' }
    ],
    sidebar: [{ text: '学习笔记', items: [{ text: '全部文章', link: '/articles/' }, ...articles] }],
    socialLinks: [{ icon: 'github', link: 'https://github.com/husichao666/cuda-learning' }],
    outline: { level: [2, 3], label: '本文目录' },
    sidebarMenuLabel: '笔记目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换浅色模式',
    darkModeSwitchTitle: '切换深色模式',
    docFooter: { prev: '上一篇', next: '下一篇' },
    search: {
      provider: 'local',
      options: {
        locales: { root: { translations: {
          button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' },
          modal: {
            noResultsText: '没有找到相关内容', resetButtonTitle: '清空搜索',
            displayDetails: '显示详细内容', backButtonTitle: '关闭搜索',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        } } }
      }
    }
  }
})
