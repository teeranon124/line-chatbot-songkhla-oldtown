# evaluate_retrieval.py
import time
import csv  # นำเข้าโมดูลจัดการไฟล์ CSV
from src.retriever import HybridRetriever

def run_dense_experiment():
    # 1. เตรียมชุดข้อมูลทดสอบพร้อม Ground Truth
    test_cases = [
        {"query": "ร้านไอติมโอ่งเปิดปิดกี่โมง?", "expected_place_id": "aitim_oang"},
        {"query": "ขอเบอร์โทรร้านสตูเกียดฟั่งหน่อย", "expected_place_id": "kiat_fang"},
        {"query": "บ้านนครในมีของเก่าอะไรให้ดูบ้าง", "expected_place_id": "baan_nakorn_in"},
        {"query": "ศาลเจ้าพ่อหลักเมืองสงขลา สร้างด้วยสถาปัตยกรรมแบบไหน", "expected_place_id": "city_pillar_shrine"}
    ]

    retriever = HybridRetriever()
    
    # ตัวแปรเก็บผลรวม
    results = {
        "Static_Top5": {"hits": 0, "total_chunks": 0, "avg_time": 0},
        "Threshold_0.6": {"hits": 0, "total_chunks": 0, "avg_time": 0},
        "Dynamic Top-K": {"hits": 0, "total_chunks": 0, "avg_time": 0}
    }

    print("🧪 กำลังรันการทดลองเปรียบเทียบกลยุทธ์ Dense Retrieval...\n" + "-"*50)

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected_place_id"]
        print(f"คำถาม {i}: '{query}' (Target: {expected})")

        # --- กลยุทธ์ 1: Static Top-K ---
        start = time.time()
        raw_hits_1 = retriever.dense.search(query, top_k=5)
        chunks_1 = [retriever.chunks[h["chunk_index"]] for h in raw_hits_1 if h["chunk_index"] < len(retriever.chunks)]
        t1 = time.time() - start
        
        hit_1 = 1 if any(c.get("place_id") == expected or expected in c.get("chunk_id", "") for c in chunks_1) else 0
        results["Static_Top5"]["hits"] += hit_1
        results["Static_Top5"]["total_chunks"] += len(chunks_1)
        results["Static_Top5"]["avg_time"] += t1

        # --- กลยุทธ์ 2: Thresholding ---
        start = time.time()
        raw_hits_2 = retriever.dense.search(query, top_k=5)
        threshold = 0.6 
        chunks_2 = [retriever.chunks[h["chunk_index"]] for h in raw_hits_2 if h["score"] > threshold and h["chunk_index"] < len(retriever.chunks)]
        t2 = time.time() - start
        
        hit_2 = 1 if any(c.get("place_id") == expected or expected in c.get("chunk_id", "") for c in chunks_2) else 0
        results["Threshold_0.6"]["hits"] += hit_2
        results["Threshold_0.6"]["total_chunks"] += len(chunks_2)
        results["Threshold_0.6"]["avg_time"] += t2

        # --- กลยุทธ์ 3: Adaptive Top-K ---
        start = time.time()
        chunks_3 = retriever.retrieve(query, mode="dense") 
        t3 = time.time() - start
        
        hit_3 = 1 if any(c.get("place_id") == expected or expected in c.get("chunk_id", "") for c in chunks_3) else 0
        results["Dynamic Top-K"]["hits"] += hit_3
        results["Dynamic Top-K"]["total_chunks"] += len(chunks_3)
        results["Dynamic Top-K"]["avg_time"] += t3

    # สรุปผล
    num_tests = len(test_cases)
    print("\n📊 สรุปผลการทดลอง (Metrics)")
    print(f"{'Strategy':<15} | {'Hit Rate (%)':<15} | {'Avg Chunks/Query':<20} | {'Avg Latency (s)'}")
    print("-" * 75)
    
    # เตรียมข้อมูลเพื่อเขียนลง CSV
    csv_data = []
    
    for name, metric in results.items():
        hit_rate = (metric['hits'] / num_tests) * 100
        avg_chunks = metric['total_chunks'] / num_tests
        avg_time = metric['avg_time'] / num_tests
        
        print(f"{name:<15} | {hit_rate:<15.1f} | {avg_chunks:<20.1f} | {avg_time:.4f}")
        
        # เพิ่มข้อมูลแต่ละแถวลงในลิสต์
        csv_data.append({
            "Strategy": name,
            "Hit Rate (%)": round(hit_rate, 2),
            "Avg Chunks/Query": round(avg_chunks, 2),
            "Avg Latency (s)": round(avg_time, 4)
        })

    # เขียนข้อมูลลงไฟล์ CSV
    csv_filename = "dense_retrieval_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Strategy", "Hit Rate (%)", "Avg Chunks/Query", "Avg Latency (s)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"\n✅ บันทึกผลลัพธ์เป็นไฟล์ CSV เรียบร้อย: {csv_filename}")

if __name__ == "__main__":
    run_dense_experiment()