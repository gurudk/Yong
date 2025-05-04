-- 插件管理器配置（Packer.nvim）
require('packer').startup(function(use)
  -- 包管理器自举
  use 'wbthomason/packer.nvim'

  use 'nvim-lua/plenary.nvim'

  -- LSP 核心
  use 'neovim/nvim-lspconfig'       -- LSP 客户端
  use 'williamboman/mason.nvim'     -- 自动安装 LSP/DAP
  use 'williamboman/mason-lspconfig.nvim'

  -- 调试器 (DAP)
  use 'mfussenegger/nvim-dap'
  use { "rcarriga/nvim-dap-ui", requires = {"mfussenegger/nvim-dap"} }
  use 'nvim-neotest/nvim-nio'
  use 'theHamsta/nvim-dap-virtual-text'

  -- 补全引擎
  use 'hrsh7th/nvim-cmp'           -- 补全核心
  use 'hrsh7th/cmp-nvim-lsp'       -- LSP 补全源
  use 'hrsh7th/cmp-buffer'         -- 缓冲区补全
  use 'hrsh7th/cmp-path'           -- 路径补全
  use 'L3MON4D3/LuaSnip'           -- 代码片段

  -- Python 专用
  use 'Vimjas/vim-python-pep8-indent'  -- 智能缩进
  use 'jeetsukumaran/vim-python-indent-black' -- Black 兼容缩进
  use {
    'linux-cultist/venv-selector.nvim',
    branch = 'regexp',
    requires = { 
      'nvim-telescope/telescope.nvim',
      'neovim/nvim-lspconfig' 
    }
  }
  -- 测试运行
  use 'vim-test/vim-test'

  -- 界面增强

  use {
    'nvim-lualine/lualine.nvim',
    requires = { 'nvim-tree/nvim-web-devicons' },
    config = function()
      require('lualine').setup({
        options = {
          theme = 'tokyonight',
          icons_enabled = true,
          component_separators = '|',
          section_separators = { left = '', right = '' },
          disabled_filetypes = {
            'alpha', 'NvimTree', 'toggleterm'
          }
        },
        sections = {
          lualine_a = {'mode'},
          lualine_b = {'branch', 'diff', 'diagnostics'},
          lualine_c = {'filename'},
          lualine_x = {'encoding', 'fileformat', 'filetype'},
          lualine_y = {'progress'},
          lualine_z = {'location'}
        }
      })
    end
  }
  use 'nvim-tree/nvim-web-devicons'-- 图标
  use 'folke/tokyonight.nvim'      -- 主题
end)

-- 初始化 Mason（自动安装 LSP/DAP）
require('mason').setup()
require('mason-lspconfig').setup({
  ensure_installed = { "pyright" }
})

-- LSP 配置
require('lspconfig').pyright.setup({
  settings = {
    python = {
      analysis = {
        autoSearchPaths = true,
        diagnosticMode = "workspace",
        typeCheckingMode = "basic"
      }
    }
  }
})

require('lspconfig').pyright.setup({
  on_attach = function(client, bufnr)
    vim.api.nvim_buf_set_option(bufnr, 'formatexpr', 'v:lua.vim.lsp.formatexpr()')
    vim.api.nvim_buf_set_keymap(bufnr, 'n', '<leader>df', '<cmd>lua vim.lsp.buf.format()<CR>', {})
  end
})

-- DAP 调试配置（Python）
local dap = require('dap')

dap.adapters.python = {
  type = 'executable',
  command = vim.fn.exepath('python3'),
  args = { '-m', 'debugpy.adapter' }
}

dap.configurations.python = {
  {
    type = 'python',
    request = 'launch',
    name = '调试当前文件（详细）',
    program = '${file}',
    pythonPath = function()
      return vim.fn.exepath('python3')  -- 明确指定解释器路径
    end,
    console = 'integratedTerminal',     -- 在终端显示输出
    internalConsoleOptions = 'neverOpen', -- 不重复打开控制台
    env = {                             -- 环境变量传递
      PYTHONPATH = vim.fn.getcwd()
    },
    justMyCode = false                   -- 允许跟踪库代码
  }
}

dap.set_log_level('TRACE')

-- 更详细的 DAP UI 配置
require('dapui').setup({
  layouts = {
    {
      elements = {
        { id = 'scopes',      size = 0.30 },  -- 变量作用域
        { id = 'breakpoints', size = 0.15 },  -- 断点列表
        { id = 'stacks',      size = 0.25 },  -- 调用栈
        { id = 'watches',     size = 0.30 }   -- 监视表达式
      },
      position = 'left',   -- 左侧面板
      size = 50            -- 宽度占比
    },
    {
      elements = {
        { id = 'repl',    size = 0.70 },  -- 调试控制台
        { id = 'console', size = 0.30 }   -- 运行日志
      },
      position = 'bottom', -- 底部面板
      size = 15            -- 高度占比
    }
  },
  force_buffers = true,
  auto_open = true,        -- 调试启动时自动打开UI
  controls = {             -- 增强控制按钮
    enabled = true,
    element = 'repl',
    icons = {
      pause = '⏸',
      play = '▶',
      step_into = '↘',
      step_over = '→',
      step_out = '↑',
      terminate = '⏹'
    }
  }
})

-- 绑定快捷键手动打开UI
vim.keymap.set('n', '<leader>du', ':lua require("dapui").toggle()<CR>')

require('luasnip').config.setup({
  enable_autosnippets = true,
  store_selection_keys = '<Tab>',
  -- 添加以下配置
  ext_opts = {
    [require('luasnip.util.types').choiceNode] = {
      active = { virt_text = { { '●', 'GruvboxOrange' } } }
    }
  }
})

-- 自动补全配置
local cmp = require('cmp')
cmp.setup({
  snippet = {
    expand = function(args) require('luasnip').lsp_expand(args.body) end,
  },
  mapping = cmp.mapping.preset.insert({
    ['<C-Space>'] = cmp.mapping.complete(),
    ['<CR>'] = cmp.mapping.confirm({ select = true }),
  }),
  sources = cmp.config.sources({
    { name = 'nvim_lsp' },
    { name = 'luasnip' },
  }, {
    { name = 'buffer' },
  })
})

-- 虚拟环境选择器
require('venv-selector').setup({
  name = { "venv", ".venv", "env" }, -- 自动检测的目录名
  auto_refresh = true,
  search_venv_managers = true,        -- 支持 Poetry/Pipenv
  dap_enabled = true                  -- 集成调试器
})

-- 主题与状态栏
vim.cmd.colorscheme('tokyonight-night')

vim.g.mapleader = " "

-- 快捷键绑定
vim.keymap.set('n', '<leader>dd', require('dapui').toggle)
vim.keymap.set('n', '<F5>', function()
  require('dap').continue()  -- 启动/继续调试
  require('dapui').open()     -- 自动打开 DAP UI
end, { desc = 'Start debugging and open UI' })

vim.keymap.set('n', '<F9>', require('dap').toggle_breakpoint)
vim.keymap.set('n', '<leader>df', ':lua vim.lsp.buf.format()<CR>')
vim.keymap.set('n', '<leader>tt', ':TestFile<CR>')

-- 必须配置项
vim.opt.encoding = 'utf-8'       -- Neovim 内部编码
vim.opt.fileencoding = 'utf-8'   -- 文件编码
vim.scriptencoding = 'utf-8'     -- 脚本编码

-- 终端兼容性增强
vim.opt.termguicolors = true     -- 启用真彩色支持
vim.opt.showmode = false         -- 禁用原生模式提示（由插件接管）

-- 绑定快捷键手动打开UI
vim.keymap.set('n', '<leader>du', ':lua require("dapui").toggle()<CR>')

vim.env.LANG = 'zh_CN.UTF-8'   -- 或 zh_CN.UTF-8
vim.env.LC_ALL = 'zh_CN.UTF-8'
