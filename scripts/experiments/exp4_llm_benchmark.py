# -*- coding: utf-8 -*-
"""
Experiment 4: Local LLM Comprehensive Evaluation Benchmark
Evaluates local open models on Ollama across:
1. qwen2.5:3b (Primary Selected Architecture)
2. gemma2:2b (Google Lightweight Open Model)
3. llama3.2:1b (Meta Edge Compact Model)
4. smollm2:1.7b (Hugging Face Compact Model)
5. qwen2.5:0.5b (Ultra-lightweight baseline)

Evaluation Dimensions:
1. Faithfulness / Hallucination Rate (% grounded strictly in retrieved context)
2. Negative Constraint Adherence (Filtering out savory food when asked for desserts)
3. Markdown & Formatting Adherence (Strict prohibition of ** and # for clean LINE UX)
4. Language Purity (Zero Chinese/foreign token leakage)
5. Inference Latency (sec) and Generation Speed (tokens/sec)
"""
import re
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    {"name": "qwen2.5:3b", "params": "3.1B", "vram_gb": 1.9, "vendor": "Alibaba / Open"},
    {"name": "gemma2:2b", "params": "2.6B", "vram_gb": 1.6, "vendor": "Google"},
    {"name": "llama3.2:1b", "params": "1.2B", "vram_gb": 1.3, "vendor": "Meta"},
    {"name": "smollm2:1.7b", "params": "1.7B", "vram_gb": 1.8, "vendor": "Hugging Face"},
    {"name": "qwen2.5:0.5b", "params": "0.5B", "vram_gb": 0.4, "vendor": "Alibaba / Open"}
]

SYSTEM_PROMPT = """คุณคือ "น้องสิงขร" ผู้ช่วยอัจฉริยะนำเที่ยวย่านเมืองเก่าสงขลา
หน้าที่ของคุณ:
1. ตอบให้ 'สั้น กระชับ ตรงประเด็น' ไม่เกิน 3-4 บรรทัด
2. ห้ามมีคำเกริ่นทักทายเยิ่นเย้อ เช่น "สวัสดีค่ะ ยินดีต้อนรับ..." หรือ "น้องสิงขรขอแนะนำ..." ให้ตอบเข้าเนื้อหาทันที
3. ห้ามใช้เครื่องหมาย Markdown เช่น เครื่องหมายดอกจัน ** หรือเครื่องหมาย # เด็ดขาด ให้ใช้ภาษาไทยธรรมดาที่เป็นธรรมชาติ
4. ต้องตอบเป็นภาษาไทยล้วน 100% ห้ามมีตัวอักษรจีนหรือภาษาต่างประเทศปะปนเด็ดขาด (เช่น ห้ามใช้คำว่า 墙壁 ให้ใช้คำว่า กำแพงหรือผนัง)
5. หากถามเรื่องของหวานหรือของกินเล่น ให้เลือกเฉพาะร้านของหวาน เช่น ร้านไอติมโอ่ง หรือบ้านขนมไทยสองแสน ห้ามนำร้านอาหารคาวมาตอบเป็นของหวาน
6. อ้างอิงข้อมูลจากบริบทอย่างเคร่งครัด หากไม่มีข้อมูลให้ตอบตามตรงว่าไม่มีข้อมูล ห้ามกุเรื่องขึ้นมาเอง"""

BENCHMARK_CASES = [
    {
        "id": "case_1_negative_constraint",
        "description": "Negative Constraint: กรองเฉพาะของหวาน ห้ามเอาอาหารคาวมาปน",
        "query": "อยากกินของหวานหรือของกินเล่นแถวนางงาม มีร้านอะไรบ้าง",
        "context": """[บริบท]:
- ร้านไอติมโอ่ง: ร้านของหวานชื่อดังบนถนนนางงาม เสิร์ฟไอศกรีมโบราณใส่โอ่งดินเผา
- ร้านแต้เฮี้ยงอิ๋ว: ร้านอาหารจีนแต้จิ๋วโบราณ เมนูเด่นคือ ข้าวต้มปลากะพง และหมูบะเต็ง
- ร้านเกียดฟั่ง: ร้านอาหารเก่าแก่ ขายข้าวสตูหมูบะเต็งและซาลาเปาลูกใหญ่""",
        "eval_type": "negative_constraint",
        "forbidden_words": ["แต้เฮี้ยงอิ๋ว", "บะเต็ง", "ข้าวสตู", "สตูหมู"],
        "required_words": ["ไอติมโอ่ง"]
    },
    {
        "id": "case_2_unanswerable_hallucination",
        "description": "Hallucination Resistance: ถามหาเบอร์โทรที่ไม่มีในบริบท ต้องตอบว่าไม่มีข้อมูล ห้ามกุเบอร์",
        "query": "ขอเบอร์โทรศัพท์ติดต่อของบ้านนครในหน่อยครับ",
        "context": """[บริบท]:
- บ้านนครใน: พิพิธภัณฑ์เอกชนบนถนนนครนอก เป็นบ้านโบราณสถาปัตยกรรมจีนผสมผสาน จัดแสดงของสะสมโบราณ เปิดให้เข้าชมฟรี""",
        "eval_type": "hallucination_check",
        "forbidden_patterns": [r"\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b", r"074\d+", r"โทรศัพท์:\s*0"],
        "required_concept": ["ไม่มีข้อมูล", "ไม่ระบุ", "ไม่ได้ระบุ", "ไม่ปรากฏ", "ไม่มีเบอร์", "ไม่มีการระบุ"]
    },
    {
        "id": "case_3_fact_verification",
        "description": "Fact Grounding: ถามข้อเท็จจริงผิด (สร้างสมัย ร.1 หรือไม่) ต้องแย้งตามบริบทว่า ร.6",
        "query": "โรงสีแดง หับ โห้ หิ้น สร้างขึ้นในสมัยรัชกาลที่ 1 ใช่หรือไม่",
        "context": """[บริบท]:
- โรงสีแดง หับ โห้ หิ้น: ก่อตั้งขึ้นเมื่อ พ.ศ. 2457 ในสมัยรัชกาลที่ 6 โดยนายสุชาติ รัตนปราการ เพื่อเป็นโรงสีข้าวพลังไอน้ำ""",
        "eval_type": "fact_grounding",
        "forbidden_words": ["ใช่แล้ว", "ใช่ครับ", "ถูกต้อง"],
        "required_words": ["รัชกาลที่ 6", "2457", "ไม่ใช่"]
    },
    {
        "id": "case_4_foreign_leakage",
        "description": "Language Purity: ประวัติโรงสีแดง ห้ามมีตัวอักษรจีนปะปนในคำตอบ",
        "query": "โรงสีแดง หับ โห้ หิ้น มีประวัติความเป็นมาอย่างไร",
        "context": """[บริบท]:
- โรงสีแดง หับ โห้ หิ้น (Hub Ho Hin - 合和兴): ภาษาฮกเกี้ยน แปลว่า ความสามัคคี ความกลมเกลียว และความเจริญรุ่งเรือง ก่อตั้ง พ.ศ. 2457 ปัจจุบันเป็นศูนย์การเรียนรู้เมืองเก่าสงขลา""",
        "eval_type": "language_purity",
        "forbidden_regex": r"[\u4e00-\u9fff]"
    },
    {
        "id": "case_5_multi_hop_conciseness",
        "description": "Multi-hop Grounding & Length: ตอบศาลหลักเมืองอยู่ถนนอะไรและตรงข้ามมีอะไร ภายใน <= 4 บรรทัด",
        "query": "ศาลเจ้าพ่อหลักเมืองสงขลาอยู่ถนนอะไร และฝั่งตรงข้ามมีร้านอะไรเด่น",
        "context": """[บริบท]:
- ศาลเจ้าพ่อหลักเมืองสงขลา: โบราณสถานศักดิ์สิทธิ์ ตั้งอยู่บนถนนนางงาม สร้างในสมัยรัชกาลที่ 3 สถาปัตยกรรมแบบจีนเก๋งจีน
- ร้านไอติมโอ่ง: ตั้งอยู่บนถนนนางงาม ฝั่งตรงข้ามศาลเจ้าพ่อหลักเมืองสงขลาพอดี""",
        "eval_type": "multi_hop_concise",
        "required_words": ["นางงาม", "ไอติมโอ่ง"],
        "max_lines": 4
    }
]


def evaluate_response(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    num_lines = len(lines)
    
    # 1. Prompt Adherence: Markdown Prohibition
    has_markdown = bool(re.search(r"(\*\*|\#\#|__|\*|_\[)", text))
    
    # 2. Prompt Adherence: Greeting Clutter
    has_greeting = bool(re.search(r"(สวัสดี|ยินดีต้อนรับ|น้องสิงขรขอแนะนำ|น้องสิงขรพร้อม)", text[:60]))
    
    # 3. Prompt Adherence: Length (<= 4 lines)
    length_ok = num_lines <= 4
    
    # 4. Chinese character leakage
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    
    # 5. Case-specific grounding & hallucination
    is_faithful = True
    c_type = case["eval_type"]
    
    if c_type == "negative_constraint":
        for fw in case["forbidden_words"]:
            if fw in text:
                is_faithful = False
                break
        if "ไอติมโอ่ง" not in text:
            is_faithful = False
    elif c_type == "hallucination_check":
        for fp in case["forbidden_patterns"]:
            if re.search(fp, text):
                is_faithful = False
                break
        has_admit = any(rc in text for rc in case["required_concept"])
        if not has_admit:
            is_faithful = False
    elif c_type == "fact_grounding":
        for fw in case["forbidden_words"]:
            if fw in text:
                is_faithful = False
                break
        has_req = any(rw in text for rw in case["required_words"])
        if not has_req:
            is_faithful = False
    elif c_type == "language_purity":
        if has_chinese:
            is_faithful = False
    elif c_type == "multi_hop_concise":
        has_req = all(rw in text for rw in case["required_words"])
        if not has_req or not length_ok:
            is_faithful = False

    prompt_score = 0
    if not has_markdown: prompt_score += 25
    if not has_greeting: prompt_score += 25
    if length_ok: prompt_score += 25
    if not has_chinese: prompt_score += 25

    return {
        "text": text,
        "num_lines": num_lines,
        "has_markdown": has_markdown,
        "has_greeting": has_greeting,
        "has_chinese": has_chinese,
        "is_faithful": is_faithful,
        "prompt_adherence_pct": prompt_score
    }


def run_benchmark():
    print("=" * 75, flush=True)
    print("🚀 Running Experiment 4: Local LLM Comprehensive Evaluation Benchmark", flush=True)
    print("=" * 75, flush=True)
    
    out_dir = Path("finalproject/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp4_llm_benchmark_results.json"
    
    results = {}
    
    for m in MODELS:
        m_name = m["name"]
        print(f"\n[EVAL] Testing Model: {m_name} (Params: {m['params']}, VRAM: {m['vram_gb']} GB)...", flush=True)
        
        case_scores = []
        total_time = 0.0
        total_tokens = 0
        
        for case in BENCHMARK_CASES:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{case['context']}\n\nคำถาม: {case['query']}"}
            ]
            
            t0 = time.time()
            try:
                res = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": m_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_predict": 128
                        }
                    },
                    timeout=25
                )
                t_elapsed = time.time() - t0
                
                if res.status_code == 200:
                    resp_json = res.json()
                    out_text = resp_json.get("message", {}).get("content", "").strip()
                    eval_metrics = evaluate_response(out_text, case)
                    eval_metrics["latency_sec"] = round(t_elapsed, 2)
                    
                    eval_count = resp_json.get("eval_count", len(out_text.split()))
                    total_tokens += eval_count
                    total_time += t_elapsed
                    
                    case_scores.append(eval_metrics)
                    print(f"  [{m_name}] Case {case['id']}: Faithful={eval_metrics['is_faithful']}, PromptAdhere={eval_metrics['prompt_adherence_pct']}%, Time={eval_metrics['latency_sec']}s", flush=True)
                else:
                    print(f"  [{m_name}] Case {case['id']}: HTTP {res.status_code}", flush=True)
            except Exception as e:
                print(f"  [{m_name}] Case {case['id']}: TIMEOUT/EXCEPTION ({e})", flush=True)
                case_scores.append({
                    "text": "[TIMEOUT / UNABLE TO RESPOND IN THAI]",
                    "num_lines": 0,
                    "has_markdown": False,
                    "has_greeting": False,
                    "has_chinese": False,
                    "is_faithful": False,
                    "prompt_adherence_pct": 0,
                    "latency_sec": 25.0
                })
                total_time += 25.0
        
        n_cases = len(case_scores)
        if n_cases > 0:
            faithfulness_rate = round(sum(1 for c in case_scores if c["is_faithful"]) / n_cases * 100, 1)
            hallucination_rate = round(100.0 - faithfulness_rate, 1)
            avg_prompt_adhere = round(sum(c["prompt_adherence_pct"] for c in case_scores) / n_cases, 1)
            markdown_leakage_pct = round(sum(1 for c in case_scores if c["has_markdown"]) / n_cases * 100, 1)
            greeting_clutter_pct = round(sum(1 for c in case_scores if c["has_greeting"]) / n_cases * 100, 1)
            chinese_leakage_pct = round(sum(1 for c in case_scores if c["has_chinese"]) / n_cases * 100, 1)
            avg_latency = round(total_time / n_cases, 2)
            tok_per_sec = round(total_tokens / total_time, 1) if total_time > 0 else 0.0
            
            results[m_name] = {
                "params": m["params"],
                "vram_gb": m["vram_gb"],
                "vendor": m["vendor"],
                "faithfulness_pct": faithfulness_rate,
                "hallucination_pct": hallucination_rate,
                "prompt_adherence_pct": avg_prompt_adhere,
                "markdown_leakage_pct": markdown_leakage_pct,
                "greeting_clutter_pct": greeting_clutter_pct,
                "chinese_leakage_pct": chinese_leakage_pct,
                "avg_latency_sec": avg_latency,
                "tokens_per_sec": tok_per_sec,
                "details": case_scores
            }
            
            # Save incremental
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Benchmark Complete! Saved to: {json_path}", flush=True)
    
    print("\n" + "=" * 90, flush=True)
    print("EXECUTIVE SUMMARY: LOCAL LLM EVALUATION BENCHMARK", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Model':<15} | {'Params':<6} | {'VRAM':<6} | {'Faithful%':<9} | {'Halluc%':<7} | {'PromptAdh%':<10} | {'MdLeak%':<7} | {'Speed(tok/s)':<12}", flush=True)
    print("-" * 90, flush=True)
    for m_name, d in results.items():
        print(f"{m_name:<15} | {d['params']:<6} | {d['vram_gb']:<4} GB | {d['faithfulness_pct']:<8}% | {d['hallucination_pct']:<6}% | {d['prompt_adherence_pct']:<9}% | {d['markdown_leakage_pct']:<6}% | {d['tokens_per_sec']:<12}", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    run_benchmark()
