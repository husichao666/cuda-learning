import { defineConfig } from 'vitepress'
import footnote from 'markdown-it-footnote'
import sidebar from './sidebar.json' with { type: 'json' }

export default defineConfig({
  lang: 'zh-CN',
  title: 'CUDA Programming Guide 中文版',
  description: 'CUDA Programming Guide 中文学习文档，按章节阅读，保留原图与术语。',
  cleanUrls: true,
  head: [['meta', { name: 'theme-color', content: '#4d7c0f' }]],
  markdown: { config: (md) => md.use(footnote) },
  themeConfig: {
    siteTitle: 'CUDA 中文指南',
    nav: [
      { text: '阅读指南', link: '/chapter-1/section-1-1', activeMatch: '/chapter-' },
      { text: '全部章节', link: '/' },
      { text: '学习笔记', link: 'https://husichao666.github.io/cuda-learning/' }
    ],
    sidebar,
    socialLinks: [{ icon: 'github', link: 'https://github.com/husichao666/cuda-learning' }],
    outline: { level: [2, 6], label: '本页小节' },
    sidebarMenuLabel: '章节目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换浅色模式',
    darkModeSwitchTitle: '切换深色模式',
    docFooter: { prev: '上一节', next: '下一节' },
    search: {
      provider: 'local',
      options: {
        locales: { root: { translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
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
