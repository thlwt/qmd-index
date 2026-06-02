#!/usr/bin/env bun
/**
 * QMD CLI - Quantized Model Database Command Line Interface
 * D:\QMD-Index\cli.js
 */

import { execSync, spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const QMD_PACKAGE_DIR = join(__dirname, 'node_modules', '@tobilu', 'qmd');
const QMD_CLI_PATH = join(QMD_PACKAGE_DIR, 'dist', 'cli', 'qmd.js');
const BUN_PATH = 'C:\\Users\\sile1618\\.bun\\bin\\bun.exe';

// Configuration paths
const CONFIG_PATH = join(__dirname, 'qmd.yml');

function runQMD(args) {
    const cmd = `${BUN_PATH} "${QMD_CLI_PATH}" ${args.join(' ')}`;
    try {
        return execSync(cmd, { encoding: 'utf8', stdio: 'inherit' });
    } catch (err) {
        console.error('❌ QMD 命令执行失败:', err.message);
        process.exit(1);
    }
}

function checkStatus() {
    console.log('🔍 检查 QMD 状态...\n');
    
    // Check Bun
    try {
        const bunVersion = execSync(`${BUN_PATH} --version`, { encoding: 'utf8' }).trim();
        console.log(`✅ Bun: v${bunVersion}`);
    } catch (err) {
        console.error('❌ Bun not found');
        process.exit(1);
    }

    // Check QMD CLI
    if (!existsSync(QMD_CLI_PATH)) {
        console.error(`❌ QMD CLI not found at ${QMD_CLI_PATH}`);
        process.exit(1);
    }
    console.log('✅ QMD CLI: OK');

    // Check Configuration
    if (!existsSync(CONFIG_PATH)) {
        console.warn(`⚠️  Config file not found at ${CONFIG_PATH}`);
    } else {
        console.log('✅ Config:', CONFIG_PATH);
        try {
            const config = readFileSync(CONFIG_PATH, 'utf8');
            console.log('   📄 YAML size:', config.length, 'bytes');
        } catch (err) {
            console.warn('⚠️  Cannot read config file');
        }
    }

    // Check Models
    const modelsDir = join(__dirname, 'models');
    if (existsSync(modelsDir)) {
        console.log(`✅ Models directory: ${modelsDir}`);
        try {
            const modelFiles = execSync(`cmd /c "dir /b "${modelsDir}"`, { encoding: 'utf8' });
            const files = modelFiles.trim().split('\n').filter(f => f && !f.startsWith('DIR'));
            console.log(`   Found ${files.length} files:`);
            files.forEach(file => console.log(`      - ${file}`));
        } catch (err) {
            console.warn('⚠️  Cannot list model files');
        }
    } else {
        console.warn('⚠️  Models directory not found');
    }

    // Check Collections
    if (existsSync(CONFIG_PATH)) {
        try {
            const config = readFileSync(CONFIG_PATH, 'utf8');
            const collections = config.match(/path:\s*(.+?)\n/g);
            if (collections) {
                console.log('\n📁 Collection Paths:');
                collections.forEach(coll => {
                    const path = coll.match(/path:\s*(.+)/)[1].trim();
                    const exists = existsSync(path.replace(/:/, '')); // Windows path check
                    console.log(`   ${exists ? '✅' : '⚠️'} ${path}`);
                });
            }
        } catch (err) {
            console.warn('⚠️  Cannot parse collection paths');
        }
    }

    console.log('\n✅ QMD Status Check Complete');
}

// Main CLI handler
const args = process.argv.slice(2);
const command = args[0];

if (!command || command === 'help' || command === '--help') {
    console.log(`
QMD CLI - Quantized Model Database Command Line Interface
==========================================================

用法：qmd <command> [options]

命令:
  search <collection> <query>   语义搜索
  list-collections              列出所有可用集合
  check-status                  检查 QMD 状态和模型加载情况
  help                          显示帮助信息

示例:
  qmd search openclaw "EvoMap Evolver"
  qmd search openclaw-memory "2026-04-14" --limit 5
  qmd list-collections
  qmd check-status

配置:
  - Collection definitions in: ${CONFIG_PATH}
  - Models directory: D:\\QMD-Index\\models\\
`);
    process.exit(0);
} else if (command === 'check-status') {
    checkStatus();
} else if (command === 'list-collections') {
    runQMD(['list-collections']);
} else if (command === 'search') {
    const collection = args[1];
    const query = args.slice(2).join(' ');
    
    if (!collection || !query) {
        console.error('❌ 用法：qmd search <collection> "<query>"');
        process.exit(1);
    }
    
    const extraArgs = [];
    for (let i = 3; i < args.length; i++) {
        if (args[i].startsWith('--')) {
            extraArgs.push(args[i]);
        }
    }
    
    runQMD(['search', collection, query, ...extraArgs]);
} else {
    console.error(`❌ 未知命令：${command}`);
    console.error('请运行 "qmd help" 查看可用命令');
    process.exit(1);
}
