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
import sys
import re
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

def 调用_LLM_分类(分类标准: str, 分类任务: str, 类目: str) -> str:
    """
    调用 Anthropic API 进行关键词分类。
    失败时返回空字符串。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""

    try:
        import urllib.request

        系统提示 = f"""{分类标准}

---
**当前任务的产品类目**: {类目}

请严格按分类标准完成上述关键词分类任务。直接输出分类结果，不要输出额外的解释或开场白。"""

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
            return body["content"][0]["text"]

    except Exception as e:
        print(f"⚠️  LLM 调用失败: {e}")
        return ""


# ============================================================
# 结果解析
# ============================================================

def 解析_LLM_结果(llm输出: str) -> list[dict]:
    """
    解析 AI 返回的分类结果。
    格式: 序号 | 关键词 | 分类 | 属性标签 | 建议动作 | 判断理由
    """
    结果 = []
    有效分类 = {"一级核心词", "二级属性词", "三级场景词", "品牌词", "竞品词", "不相关/宽泛词"}

    for line in llm输出.strip().split("\n"):
        line = line.strip()
        # 跳过空行、表格分隔线、非数据行
        if not line or line.startswith("---") or line.startswith("序号") or line.startswith("```"):
            continue
        # 按 | 分割
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        # 跳过序号为纯数字开头的非数据行
        try:
            序号 = parts[0]
            if not 序号.isdigit():
                continue
        except ValueError:
            continue

        分类 = parts[2] if len(parts) > 2 else ""
        # 标准化分类名（AI 可能输出略有差异）
        for 标准分类 in 有效分类:
            if 标准分类 in 分类:
                分类 = 标准分类
                break

        结果.append({
            "关键词": parts[1] if len(parts) > 1 else "",
            "分类": 分类 if 分类 in 有效分类 else "不相关/宽泛词",
            "属性标签": parts[3] if len(parts) > 3 else "-",
            "建议动作": parts[4] if len(parts) > 4 else "待判定",
        })

    return 结果


# ============================================================
# 主流程
# ============================================================

def 处理关键词表(输入_csv路径: str, 输出_csv路径: str, 产品类目: str = "家居摆件-金色系") -> dict:
    """读取关键词 CSV → 构造 Prompt → 调 LLM 分类 → 输出结果"""

    # 读取关键词
    with open(输入_csv路径, encoding="utf-8") as f:
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
    if llm结果:
        分类数据 = 解析_LLM_结果(llm结果)
        print(f"✅ AI 分类完成，获取 {len(分类数据)} 条结果")
    else:
        print("⚠️  未检测到 ANTHROPIC_API_KEY，输出 Prompt 预览（前 1500 字符）:\n")
        print("=" * 60)
        print(分类任务[:1500])
        print("=" * 60)
        print("\n💡 设置 ANTHROPIC_API_KEY 环境变量后重新运行即可调用 AI 分类。")
        return {"总词数": len(raw_rows), "去重后": len(seen), "提示": "未调用LLM"}

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
            "搜索量": info["搜索量"],
            "竞争度": info["竞争度"],
            "PPC竞价(USD)": info["PPC竞价(USD)"],
        })

    # 写入输出 CSV
    输出列 = ["关键词", "分类", "属性标签", "词频", "建议动作", "搜索量", "竞争度", "PPC竞价(USD)"]
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
