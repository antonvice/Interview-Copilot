# FREE TO USE FULLY LOCAL Interview Copilot

Interview Copilot is an advanced web application designed to assist with conducting professional interviews by providing real-time transcription and generating insightful suggestions based on the ongoing conversation.

![demo](https://github.com/antonvice/Real-Time-Free-Local-Interview-Copilot/blob/master/1.gif)

## To DO

* Add Model selection
* Speaker diarization, split on different roles


## Features

- **Automatic Speech Recognition**: Utilizes the `distil-whisper/distil-large-v2` model for efficient and accurate real-time transcription of interviews.
- **Intelligent Suggestion Generation**: Leverages the power of `ollama` to generate context-aware suggestions and questions to further explore the interviewee's responses.
- **Interactive Web Interface**: Provides a user-friendly interface for recording interviews, viewing transcriptions, and receiving suggestions, all in real-time.

## Installation

Before you start, make sure you have Python 3.8+ and pip installed. You will also need `torch` installed in your environment to use the `transformers` library.

1. **Clone the repository:**

```bash
git clone https://github.com/antonvice/Real-Time-Free-Local-Interview-Copilot
cd Real-Time-Free-Local-Interview-Copilot
```

2. **Install dependencies:**
```
bash
pip install fastapi uvicorn transformers tempfile ollama
```

3. **Run the application:**
```
bash
uvicorn app:app --reload
The application will be available at http://127.0.0.1:8000.
```

4. **Usage**
* Navigate to http://127.0.0.1:8000 in your web browser to access the Interview Copilot interface.
* Click on "Start Recording" to begin recording the interview. The transcription will appear in real-time as you proceed with the interview.
* Click on "Get Suggestion" to receive context-aware suggestions based on the current conversation.
* Use the provided suggestions to guide the direction of the interview or to explore topics in more depth.

## Technology Stack

* Backend: FastAPI for handling API requests and serving the web application.
* Frontend: Tailwind CSS for styling and JavaScript for handling user interactions and AJAX requests.
* ASR Model: distil-whisper/distil-large-v2 from Hugging Face's transformers library for automatic speech recognition.
* SLM: ollama with gemma:2b-i for generating insightful and context-aware suggestions.

## Contributing

Contributions to Interview Copilot are welcome! Please refer to the project's issues tab on GitHub to find areas where you can help. Before making contributions, please read our contributing guidelines to ensure a smooth collaboration process.

## License

Interview Copilot is open-source software licensed under the MIT license. See the LICENSE file for more details.

## Acknowledgments

Special thanks to [AI Jason](https://github.com/JayZeeDesign) for paving the path and [NamelyAI](https://namelyai.com) for giving me the skills to accomplish this project
