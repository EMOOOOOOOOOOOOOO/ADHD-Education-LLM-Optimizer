import os
import sys
import csv
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# GPT 初始化
client = OpenAI(
    api_key="Fill Ur API Here",
    default_headers={"User-Agent": "gpt-text-optimizer"}
)

def build_prompt():
    return f"""
    你是一個專門協助減輕 ADHD 或 ASD 學生閱讀與認知負荷的專業語言輔助系統。  
你需要分析以下提供的文章內容，並完成以下任務：
分析以下文章內容：


任務：
1. 列出文章中可能造成 ADHD 或 ASD 學生認知或閱讀負荷的「較難詞彙」，並針對每個詞彙，提供至少二～五個難度介於國小一年級～高中三年級的「簡單詞彙」建議取代。

2. 找出文章中可能造成閱讀負荷的「較長句子」，並針對每個句子，提供三～四種較短且容易理解（難度介於國小一年級～高中三年級）的「短句」版本。

3. 這是最重要的一點！！！
你身為輔助系統，同時要考慮到正常學生可能會有關於公平性的疑慮，你如果要提供長句變為短句，要保持"同類異構"、"題目內容不能只講到題目要考的觀念，只能簡化：引導到題目觀念的攏長之題目敘述"、"我們只是要簡化題目敘述！但每個考的觀念要一樣"

4. 雖然要部分簡化句子或是詞彙，但是要保證替換後語意順暢並且方便閱讀理解，如果你收到的內容有部分錯字，請猜測。

請以以下格式回傳結果（務必遵守）：

詞彙簡化建議：
- "表面摩擦力極小" 建議換為: "很滑，幾乎沒有摩擦力" or "滑溜溜的" or "不會卡住"
- "濕度大" 建議換為: "很潮濕" or "水氣很多" or "空氣濕濕的"
- "明顯" 建議換為: "清楚" or "很清楚" or "看得出來"
- "地景因應式農業聚落" 建議換為: "配合地形的農村" or "因地制宜的農業社區" or "適應環境的農業村落"

句子簡化建議：
-第1題

- 原句："位於季風氣候明顯地區的甲鎮，年降雨集中於夏季，氣溫高且濕度大。該地居民利用此氣候條件種植熱帶水果，並發展出「地景因應式農業聚落」。隨著農產品銷量增加，地方政府擬提升物流效率。"  
  建議替代為："甲鎮是個夏天會下很多雨，氣溫又高濕度大的地方。這裡的人善於利用這種天氣，種了很多熱帶水果。因為農產品賣了不少，地方政府想要讓運送農產品變得更快更好。"

-第2題

- 原句："由於全球暖化造成的氣候變遷，導致極端天氣事件頻率增加，對農業生產系統產生重大衝擊，農民必須調整種植策略以適應環境變化。"  
  建議替代為："因為地球變熱，天氣變得很奇怪，常常有很強的風雨。這讓種田變得很困難，農夫們要改變種植的方法。"

- 原句："該實驗利用不同濃度的化學溶液，觀察其對植物生長速率的影響，並記錄相關數據進行統計分析。"  
  建議替代為："這個實驗用不同濃度的化學藥水，看看對植物長大的速度有什麼影響，然後把觀察到的數字記錄下來分析。"
""".strip()


def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

@app.route('/api/analyze_text', methods=['POST'])
def analyze_text_api():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Text is empty'}), 400
        
        prompt = build_prompt(text)
        result = call_gpt(prompt)
        
        # 解析GPT回應，提取詞彙和句子建議
        suggestions = parse_gpt_response(result)
        
        return jsonify({
            'success': True,
            'raw_response': result,
            'suggestions': suggestions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_gpt_response(response):
    """解析GPT回應，提取詞彙和句子建議"""
    lines = response.split('\n')
    word_suggestions = []
    sentence_suggestions = []
    
    current_section = None
    current_sentence = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if '詞彙簡化建議' in line:
            current_section = 'words'
        elif '句子簡化建議' in line:
            current_section = 'sentences'
        elif line.startswith('- 原詞：') and current_section == 'words':
            # 解析詞彙建議格式
            try:
                parts = line.split('|')
                if len(parts) >= 4:
                    original = parts[0].replace('- 原詞："', '').replace('"', '').strip()
                    before_context = parts[1].replace('前5字："', '').replace('"', '').strip()
                    after_context = parts[2].replace('後5字："', '').replace('"', '').strip()
                    suggestions_text = parts[3].replace('建議："', '').replace('"', '').strip()
                    suggestions_list = [s.strip().replace('"', '') for s in suggestions_text.split(' or ')]
                    
                    word_suggestions.append({
                        'original': original,
                        'before_context': before_context,
                        'after_context': after_context,
                        'suggestions': suggestions_list
                    })
            except Exception as e:
                print(f"解析詞彙建議錯誤: {e}")
                
        elif line.startswith('- 題號：') and current_section == 'sentences':
            # 開始新的句子建議
            if current_sentence:
                sentence_suggestions.append(current_sentence)
            current_sentence = {'question_number': line.replace('- 題號：', '').strip()}
            
        elif line.startswith('- 原句：') and current_section == 'sentences' and current_sentence:
            original_sentence = line.replace('- 原句："', '').replace('"', '').strip()
            current_sentence['original'] = original_sentence
            
        elif line.startswith('- 前5字：') and current_section == 'sentences' and current_sentence:
            parts = line.split('|')
            if len(parts) >= 2:
                before_context = parts[0].replace('- 前5字："', '').replace('"', '').strip()
                after_context = parts[1].replace('後5字："', '').replace('"', '').strip()
                current_sentence['before_context'] = before_context
                current_sentence['after_context'] = after_context
                
        elif line.startswith('- 建議：') and current_section == 'sentences' and current_sentence:
            suggested = line.replace('- 建議："', '').replace('"', '').strip()
            current_sentence['suggested'] = suggested
    
    # 添加最後一個句子建議
    if current_sentence and current_section == 'sentences':
        sentence_suggestions.append(current_sentence)
    
    return {
        'words': word_suggestions,
        'sentences': sentence_suggestions
    }

def main():

    print("傳送至 GPT 分析中...\n")
    prompt = build_prompt()
    result = call_gpt(prompt)

    print("\nGPT 分析結果：\n")
    print(result)

if __name__ == "__main__":
    if 2 > 1:
        main()
    else:
        print("啟動 Flask 服務器...")
        app.run(debug=True, port=8000)
