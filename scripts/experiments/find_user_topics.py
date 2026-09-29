import json

transcript_path = r'C:\Users\teera\.gemini\antigravity\brain\4a3fbd9a-1650-44c1-b8b9-2a6c0c4d707f\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if any(k in content for k in ['คนที่', 'ส่วนของผม', 'เล่า', 'พูด', 'สเตป', 'หัวข้อ']):
                print(f"--- Step {data.get('step_index')} ---")
                print(content)
                print()
