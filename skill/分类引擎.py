#!/usr/bin/env python3
"""
关键词分类引擎 —— LLM 驱动版本

将分类标准（skill/分类标准.md）+ 关键词列表发给 AI 进行分类判断。
AI 天然理解语义——无需手工维护词库，切换类目只需在 Prompt 中说明产品类目即可。

本地运行需要 ANTHROPIC_API_KEY 环境变量。
若无 API Key，脚本会输出即将发送给 AI 的 Prompt 预览。
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SKILL_DIR = Path(__file__).parent

# ============================================================
# 加载分类标准
# ============================================================

def 加载分类标准() -> str:
    """读取分类标准 Prompt 文档"""
    标准文件 = SKILL_DIR / "分类标准.md"
    if not 标准文件.exists():
        print(f"❌ 找不到分类标准文件: {标准文件}")
        sys.exit(1)
    return 标准文件.read_text(encoding="utf-8")


# ============================================================
# Prompt 构造
# ============================================================

def 构造分类任务_prompt(关键词列表: list[str], 产品类目: str) -> str:
    """
    构造发给 AI 的分类任务 prompt。
    将分类标准中的示例类目替换为用户指定的类目。
    """
    return f"""请对以下 {len(关键词列表)} 个关键词进行分类。

**产品类目**: {产品类目}

**要求**: 对每个关键词，按分类标准判断其属于：一级核心词 / 二级属性词 / 三级场景词 / 品牌词 / 竞品词 / 不相关/宽泛词。

**关键词列表**:
{chr(10).join(f"{i+1}. {kw}" for i, kw in enumerate(关键词列表))}

**输出格式**（严格按此格式，每行一个，不要输出多余内容）:
```
序号 | 关键词 | 分类 | 属性标签 | 建议动作 | 判断理由
```

示例输出:
```
1 | 蓝牙耳机 | 一级核心词 | - | 仅Listing嵌入 | 基础品类词，无属性修饰
2 | 防水蓝牙耳机 降噪 | 二级属性词 | 功能:防水,降噪 | 建议投放 | 品类词+2个功能属性
3 | 熬夜急救眼霜 | 三级场景词 | 场景:熬夜; 痛点:急救 | 建议投放 | 含痛点信号词"急救"+场景词"熬夜"
```"""


# ============================================================
# LLM 调用
# ============================================================

def 调用_LLM_分类(分类标准: str, 分类任务: str, 类目: str):
    """
    调用 Anthropic API 进行关键词分类。
    返回: 成功返回 AI 输出文本，无 API Key 返回 None，API 失败返回 ""
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # 无 Key

    max_重试 = 2
    for attempt in range(max_重试 + 1):
        try:
            系统提示 = 分类标准 + "\n\n---\n**当前任务的产品类目**: " + 类目 + "\n\n请严格按分类标准完成上述关键词分类任务。直接输出分类结果，不要输出额外的解释或开场白。"

            请求体 = json.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 4096,
                "system": 系统提示,
                "messages": [
                    {"role": "user", "content": 分类任务}
                ]
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=请求体,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # C1: 检查 content 长度和 block type
            content_list = body.get("content", [])
            if not content_list:
                raise ValueError("API 返回空 content 列表")

            text_blocks = [
                b.get("text", "")
                for b in content_list
                if b.get("type") == "text"
            ]
            if not text_blocks:
                raise ValueError(f"API 返回无 text 类型 block，实际类型: {[b.get('type') for b in content_list]}")

            return "\n".join(text_blocks)

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < max_重试:
                wait = 2 ** attempt  # 1s, 2s
                print(f"⚠️  API 调用失败（第{attempt+1}次），{wait}秒后重试: {e}")
                time.sleep(wait)
            else:
                print(f"❌ API 调用失败（已重试{max_重试}次）: {e}")
        except Exception as e:
            print(f"❌ API 调用异常: {e}")
            break  # 非网络错误，不重试

    return ""  # API 失败


# ============================================================
# 结果解析
# ============================================================

def 解析_LLM_结果(llm输出: str) -> list[dict]:
    """
    解析 AI 返回的分类结果（6 列 | 分隔格式）。
    """
    结果 = []
    有效分类 = {"一级核心词", "二级属性词", "三级场景词", "品牌词", "竞品词", "不相关/宽泛词"}
    有效建议动作 = {"建议投放", "仅Listing嵌入", "建议否定", "待判定"}

    for line in llm输出.strip().split("\n"):
        line = line.strip()
        # 跳过空行、表格分隔线、表头行、markdown 代码块标记
        if not line or line.startswith("---") or line.startswith("序号") or line.startswith("```"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:       # N1: 6 列
            continue

        # C5: 直接判断 isdigit，无 try/except
        序号 = parts[0]
        if not 序号.isdigit():
            continue

        # C2: 先剥离括号注释，再精确匹配
        raw_分类 = parts[2] if len(parts) > 2 else ""
        分类_标准化 = re.sub(r'[（(][^)）]*[)）]', '', raw_分类).strip()
        分类 = 分类_标准化 if 分类_标准化 in 有效分类 else ""

        # C10: 建议动作校验
        raw_建议动作 = parts[4].strip() if len(parts) > 4 else ""
        建议动作 = raw_建议动作 if raw_建议动作 in 有效建议动作 else "待判定"

        结果.append({
            "关键词": parts[1] if len(parts) > 1 else "",
            "分类": 分类 if 分类 else "不相关/宽泛词",
            "属性标签": parts[3] if len(parts) > 3 else "-",
            "建议动作": 建议动作,
            "判断理由": parts[5] if len(parts) > 5 else "",   # N2
        })

    # N5: 检测 LLM 对同关键词输出多行
    已见 = {}
    for r in 结果:
        kw_lower = r["关键词"].lower()
        if kw_lower in 已见:
            existing = 已见[kw_lower]
            if existing["分类"] != r["分类"]:
                print(f"⚠️  关键词「{r['关键词']}」存在冲突分类: \"{existing['分类']}\" vs \"{r['分类']}\"，已保留首次")
            else:
                print(f"⚠️  关键词「{r['关键词']}」重复输出（相同分类），已合并")
        else:
            已见[kw_lower] = r

    return list(已见.values())


# ============================================================
# 主流程
# ============================================================

def 处理关键词表(输入_csv路径: str, 输出_csv路径: str, 产品类目: str = "家居摆件-金色系") -> dict:
    """读取关键词 CSV → 构造 Prompt → 调 LLM 分类 → 输出结果"""

    # 读取关键词
    with open(输入_csv路径, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # 去重 + 词频统计
    seen = {}
    for row in raw_rows:
        kw = row["关键词"].strip().lower()
        if kw not in seen:
            seen[kw] = {
                "关键词": row["关键词"].strip(),
                "搜索量": row.get("搜索量", ""),
                "竞争度": row.get("竞争度", ""),
                "PPC竞价(USD)": row.get("PPC竞价(USD)", ""),
                "词频": 1
            }
        else:
            seen[kw]["词频"] += 1

    去重关键词 = list(seen.keys())

    # 加载分类标准
    分类标准 = 加载分类标准()

    # 构造 Prompt
    分类任务 = 构造分类任务_prompt(去重关键词, 产品类目)

    # 尝试调 LLM
    print(f"🤖 正在通过 AI 对 {len(去重关键词)} 个关键词进行分类...")
    llm结果 = 调用_LLM_分类(分类标准, 分类任务, 产品类目)

    分类数据 = []
    if llm结果 is None:
        # C6: 无 API Key
        print("⚠️  未配置 ANTHROPIC_API_KEY，输出 Prompt 预览（前 1500 字符）:\n")
        print("=" * 60)
        print(分类任务[:1500])
        print("=" * 60)
        print("\n💡 设置 ANTHROPIC_API_KEY 环境变量后重新运行即可调用 AI 分类。")
        return {"总词数": len(raw_rows), "去重后": len(seen), "提示": "未调用LLM"}
    elif llm结果 == "":
        # C6: API 调用失败
        print("❌ AI 分类失败（已重试 2 次），请检查网络或密钥有效性后重试。")
        return {"总词数": len(raw_rows), "去重后": len(seen), "提示": "API调用失败"}
    else:
        分类数据 = 解析_LLM_结果(llm结果)
        print(f"✅ AI 分类完成，获取 {len(分类数据)} 条结果")

    # 合并分类结果到原始数据
    统计 = {
        "总词数": len(raw_rows),
        "去重后": len(seen),
        "一级核心词": 0, "二级属性词": 0, "三级场景词": 0,
        "品牌词": 0, "竞品词": 0, "不相关/宽泛词": 0, "建议否定词": 0,
    }

    输出行列表 = []
    分类映射 = {r["关键词"].lower(): r for r in 分类数据}

    for kw, info in seen.items():
        分类信息 = 分类映射.get(kw, {"分类": "不相关/宽泛词", "属性标签": "-", "建议动作": "待判定"})
        分类 = 分类信息["分类"]
        属性标签 = 分类信息["属性标签"]
        建议动作 = 分类信息["建议动作"]

        if 分类 in 统计:
            统计[分类] += 1
        if 建议动作 == "建议否定":
            统计["建议否定词"] += 1

        输出行列表.append({
            "关键词": info["关键词"],
            "分类": 分类,
            "属性标签": 属性标签,
            "词频": str(info["词频"]),
            "建议动作": 建议动作,
            "判断理由": 分类信息.get("判断理由", ""),
            "搜索量": info["搜索量"],
            "竞争度": info["竞争度"],
            "PPC竞价(USD)": info["PPC竞价(USD)"],
        })

    # 写入输出 CSV
    输出列 = ["关键词", "分类", "属性标签", "词频", "建议动作", "判断理由",
              "搜索量", "竞争度", "PPC竞价(USD)"]
    with open(输出_csv路径, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=输出列)
        writer.writeheader()
        writer.writerows(输出行列表)

    return 统计


def 生成群消息摘要(统计: dict, 类目名: str) -> str:
    """根据统计结果生成钉钉群消息摘要"""
    if "提示" in 统计:
        return f"⚠️  关键词整理预览 | {类目名}\n\n总词数：{统计['总词数']} → 去重后：{统计['去重后']}\n\n请设置 ANTHROPIC_API_KEY 后重新运行。"

    有效词 = 统计["去重后"] - 统计.get("不相关/宽泛词", 0)
    return f"""📊 关键词整理完成 | {类目名}

━━ 统计 ━━
总词数：{统计['总词数']} → 去重后：{统计['去重后']} → 有效词：{有效词}

📌 一级核心词：{统计.get('一级核心词', 0)} 个（品类认知类）
🎯 二级属性词：{统计.get('二级属性词', 0)} 个（主力投放）
💰 三级场景词：{统计.get('三级场景词', 0)} 个（高转化）
🏷 品牌/竞品词：{统计.get('品牌词', 0) + 统计.get('竞品词', 0)} 个
✂ 建议否定词：{统计.get('建议否定词', 0)} 个"""


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    输入文件 = SKILL_DIR / "测试" / "示例关键词数据.csv"
    输出文件 = SKILL_DIR / "测试" / "分类结果.csv"
    类目 = "家居摆件-金色系"

    if len(sys.argv) > 1:
        类目 = sys.argv[1]

    if not 输入文件.exists():
        print(f"❌ 找不到输入文件: {输入文件}")
        sys.exit(1)

    统计 = 处理关键词表(str(输入文件), str(输出文件), 类目)
    摘要 = 生成群消息摘要(统计, 类目)

    print("\n" + 摘要)
    if "提示" not in 统计:
        print(f"\n✅ 完整结果已输出至: {输出文件}")
