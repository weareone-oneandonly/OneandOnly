from flask import Flask, request, render_template
from langchain.prompts import PromptTemplate
import os
from openai import AzureOpenAI
import json
from PIL import Image
import requests
from io import BytesIO
import base64

app = Flask(__name__)

client = AzureOpenAI(
        api_version="2024-05-01-preview",
        azure_endpoint="https://aoai-12-pjy.openai.azure.com/",
        api_key= '04d3448ace934b12a198fc35e3759135',
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    template = '''
    {theme} 분위기의 도시락 케이크 디자인을 위에서 바라본 모습. 케이크는 부드러운 {color}이 섞인 버터크림으로 장식되어 있으며, {character}이/가 {position}해 있고 케이크의 가장자리에 이를 반영한 장식이 있습니다. 전체적인 디자인은 깔끔하고 단순하며 명확함에 중점을 둡니다. 도시락 케이크의 모양은 {shape}입니다. 도시락 케이크 지름은 {size}입니다. 사진은 오버헤드뷰로 보입니다.
    '''

    prompt = PromptTemplate(
    input_variables=["theme", "color", "character", "position", "shape", "size"],
    template=template,
    )

    theme = request.form['theme']
    color = request.form['color']
    character = request.form['character']
    position = request.form['position']
    shape = request.form['shape']
    size = request.form['size']

    prompt_string = prompt.format(theme=theme, color=color, character=character, position=position, shape=shape, size = size)

    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt_string,
        n=1,
    )

    image_data = json.loads(result.model_dump_json())
    image_url = image_data['data'][0]['url']

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 가격 책정
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", 
                 "text": "사진 속 케이크를 제작하기 위해 필요한 가격을 책정해줘."},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
    )
    predicted_price = response.choices[0].message.content

    return render_template('result.html', img_str = img_str, predicted_price = predicted_price)

@app.route('/modify', methods=['POST'])
def modify():

    image_url = request.form['image_url']
    color = request.form['color']
    size = request.form['size']

    prompt_content = f"Please write a prompt in Korean that describes this photo  describing a cake in as much detail as possible so that dall-e3 can turn it into a cake design. The main color of the design is {color}. And the size is {size}. Please create it with a feeling and design as similar to the photo as possible. And the photo is overhead view. This is really important to me, so I kindly ask you to be precise. If what I want is accurately captured, I'll tip you whatever you want."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_content},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
    )

    response_text = response.choices[0].message.content

    print(response_text)

    # 시작과 끝을 나타내는 구분자 설정
    start_marker = '---'
    end_marker = '---'

    # 시작점과 끝점 찾기
    start_index = response_text.find(start_marker)
    end_index = response_text.find(end_marker, start_index + len(start_marker))

    # 시작점과 끝점 사이의 텍스트를 슬라이싱
    if start_index != -1 and end_index != -1:
        generated_prompt = response_text[start_index + len(start_marker):end_index].strip()
    else:
        generated_prompt = "구분자를 찾을 수 없습니다."
    print(generated_prompt)

    # 프롬프트로 이미지 요청하기
    result = client.images.generate(
        model="dall-e-3",
        prompt=generated_prompt,
        n=1,
    )

    image_data = json.loads(result.model_dump_json())
    image_url = image_data['data'][0]['url']

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    ### 가격 책정 ###
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", 
                 "text": """
                 Please calculate the price needed to create the cake in the photo. Provide the details in the following HTML format:
                    <h3>Cake Price Estimate</h3>
                    <ul>
                        <li><strong>재료 비용</strong>: Describe the costs for ingredients such as sponge, icing, coloring, decorative flowers, etc.</li>
                        <li><strong>노동 시간</strong>: Estimate the cost based on the time required to make the cake. Consider more time for complex designs.</li>
                        <li><strong>필요한 장비</strong>: Include any costs for the tools and equipment used in the creation.</li>
                        <li><strong>디자인 복잡성</strong>: Adjust the price based on the complexity of the design.</li>
                        <li><strong>시장 가격</strong>: Compare the price with similar cakes in the market to ensure competitiveness.</li>
                    </ul>
                    <p><strong>최종 책정 가격</strong>: Provide a total estimated price range in local currency.</p>
                """},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
    )
    predicted_price = response.choices[0].message.content

    return render_template('result.html', img_str = img_str, predicted_price = predicted_price)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000, debug=True)