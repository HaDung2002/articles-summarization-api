# Serve Mistral 7B with FastAPI using Colab GPU and Ngrok for Article Summarization

This project demonstrates how to serve the Mistral 7B model, an open-source large language model (LLM) by MistralAI, using FastAPI on Google Colab. Ngrok is used to expose the local web server running on Colab to a public URL. The application processes requests to summarize articles in Korean.

## Directory Structure

```plaintext
articles-summarization-api/
├── Mistral-7B-Instruct-v0.3-Q4_K_M.gguf     # Pre-trained model file
├── README.md                                # Project documentation
├── requirements.txt                         # List of Python dependencies
├── run.py                                   # Script to construct prompt and make API request
├── server.py                                # FastAPI server implementation
├── summary.py                               # Function to make HTTP request to the server
```

## Prerequisites

Before running the project, ensure you have the following:

1. A Google Colab account.
2. Ngrok account and authtoken (required to expose Colab to a public URL).
3. The Mistral 7B model GGUF file.

## Setup

### Step 1: Clone the repository

Clone this repository to your local machine or directly to Google Colab.

### Step 2: Download Mistral 7B model file

Download the `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` model using the follow commands

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF --include "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" --local-dir . --local-dir-use-symlinks False
```

### Step 3: Install dependencies

Install the required Python packages listed in `requirements.txt`. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Step 4: Set up Ngrok

1. Sign up for an Ngrok account if you haven't already.
2. Retrieve your authtoken from the [Ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken "Ngrok dashboard").
3. Add your Ngrok authtoken in the Colab notebook or the environment:

```bash
!ngrok authtoken <YOUR_AUTHTOKEN>
```

### Step 5: Run the Server

Run the FastAPI server using the provided Colab notebook or locally. This will start the server and expose it using Ngrok.

```bash
python server.py
```

This will print a public URL where the server is accessible, for example:

```plaintext
Public URL: https://<random-string>.ngrok-free.app
```

### Step 6: Test the Summarization

1. Use the `run.py` script to send a summarization request. Update the `construct_prompt` function and `CONTENT` variable with your specific data, and replace the placeholder URL with the public URL provided by Ngrok.

```python
from summary import summary

def construct_prompt(content):
    # Create the prompt for summarization
    prompt = f"""
    ### Article:
    {content}

    제시되는 기사 내용을 다음과 같은 조건에 맞춰 요약해줘
    - 요약은 3가지 이내로 정리
    - 기사 내용이 짧은 경우 요약 항목은 3개 보다 작아도 됨
    - 요약에 대한 상세 설명은 작성하지 않고 요약 항목에 함께 짧은 문장으로 구성
    - 인터뷰 대상을 인물/기관으로 정리
    - 인터뷰 대상은 주요 인물만 정리해줘
    - 요약 내용 위에 제목을 표시해 주세요.
    - 기사 작성자를 보여주세요.

    ### Summary (Korean):
    """
    return prompt

CONTENT = """
"연봉은 6억"…무명 개그맨, 유튜버로 '대박' 낸 비결은
입력2024.06.21. 오전 10:00  수정2024.06.21. 오전 10:21 기사원문
[파이낸셜뉴스] KBS 공채 개그맨으로 데뷔했지만 무명시절을 보낸 정승빈이 유튜브 및 기타 사업으로 연봉이 5억~6억원에 달한다고 근황을 전했다.
지난 19일 유튜브 채널 '황예랑'에는 '월 5천만원씩 벌어도 더 악착같이 모으고 아끼는 이유'라는 제목의 영상이 게재됐다.
영상에서 정승빈은 자신을 32세 개그맨이자 유튜브 크리에이터, 자영업자로 소개했다. 정승빈은 구독자 82만명을 보유한 유튜브 채널 '깨방정'을 운영 중이다.
정승빈은 유튜브를 시작하게 된 계기에 대해 "대부분의 개그맨은 2022년 '개그콘서트'가 폐지된 이후 유튜브를 시작했다. 나는 그 이전인 2018년 다른 무명 개그맨 친구를 따라서 유튜브를 하게 됐다"고 말했다.
그는 유튜브를 통해 연봉으로 5억~6억원 정도 번다며 "저축은 한달에 못 해도 3000만원 정도는 한다. 생활비 300만~400만원 정도 빼고 무조건 저축한다"고 밝혔다. 생활비를 주로 지출하는 영역은 배달 음식과 운동이라고 부연했다
그러면서 "경제적 여유가 생긴다면 내가 좋아하는 개그맨을 취미로 하고 싶다"며 "성공 비결은 항상 위기의식을 갖는 것"이라고 덧붙였다.
정승빈은 영상이 공개된 후에도 댓글을 통해 "훌륭하신 분들이 훨씬 많은데 제가 이런 영상을 찍어도 되나 많이 민망하긴 하다. 다들 많이 버시고 돈도 지키시고 건강도 지키시길 바란다"고 겸손한 모습을 보였다.
정승빈은 구독자 82만명을 보유한 유튜브 채널 '깨방정'을 운영하고 있다. 지난 5일 2024 한류 인플루언서 대상 어워즈에서 크리에이터 대상을 받았다. 2020년엔 유튜브 코리아 올해의 핫 채널 코미디 부문 탑(Top)2에 올랐다.
#KBS #유튜버 #사업 #정승빈 #공채 개그맨 #깨방정
김주리 기자 (rainbow@fnnews.com)
"""

PROMPT_SAMPLE = construct_prompt(CONTENT)

# Define the API endpoint
url = "https://<public-url-from-ngrok>/chat/summary"

# Create the payload
payload = {
    "messages": [{"role": "user", "content": PROMPT_SAMPLE}],
    "model": "Mistral-7B-Instruct-v0.3",
    "max_tokens": 1024,
    "temperature": 0.7
}

# Set the headers
headers = {
    "Content-Type": "application/json"
}

summary(url, payload, headers)
```

2. Run `run.py` to make a summarization request and view the response.

## Troubleshooting

- **Model Not Loading**: Ensure the Mistral model GGUF file are correctly placed and paths are set properly.
- **Ngrok Connection Issues**: Verify your authtoken and network connection. Ensure no firewall is blocking Ngrok.
- **Dependencies Issues**: Check if all dependencies in `requirements.txt` are installed. Re-run the pip install command if needed.

## Contributing

Feel free to fork this repository and create pull requests to contribute. For major changes, please open an issue to discuss what you would like to change.

## Acknowledgments

- [bartowski](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF) for providing the Mistral 7B GGUF model.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [Ngrok](https://ngrok.com/) for the tunneling service.
