try:
    from camel.toolkits import AudioAnalysisToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool

class AskQuestionAboutAudioCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(AudioAnalysisToolkit().ask_question_about_audio)
        )

class Audio2TextCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(AudioAnalysisToolkit().audio2text)
        )

if __name__ == "__main__":
    # Test Audio2TextCamel
    audio_file = "rllm-internal/rllm/data/train/web/gaia_files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"
    print("Testing Audio2TextCamel with file:", audio_file)
    result = Audio2TextCamel()(**{"audio_path": audio_file})
    print("Audio2TextCamel result:", result)

    # Test AskQuestionAboutAudioCamel
    question = "What is the main topic being discussed?"
    print("\nTesting AskQuestionAboutAudioCamel with file:", audio_file)
    print("Question:", question)
    result = AskQuestionAboutAudioCamel()(**{
        "audio_path": audio_file,
        "question": question
    })
    print("AskQuestionAboutAudioCamel result:", result)

