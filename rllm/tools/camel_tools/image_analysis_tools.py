try:
    from camel.toolkits import ImageAnalysisToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool

class ImageToTextCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(ImageAnalysisToolkit().image_to_text)
        )

class AskQuestionAboutImageCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(ImageAnalysisToolkit().ask_question_about_image)
        )

if __name__ == "__main__":
    # Test ImageToTextCamel
    image_file = "rllm-internal/rllm/data/train/web/gaia_files/b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg"
    print("Testing ImageToTextCamel with file:", image_file)
    result = ImageToTextCamel()(**{"image_path": image_file})
    print("ImageToTextCamel result:", result)

    # Test AskQuestionAboutImageCamel
    question = "What is shown in this image?"
    print("\nTesting AskQuestionAboutImageCamel with file:", image_file)
    print("Question:", question)
    result = AskQuestionAboutImageCamel()(**{
        "image_path": image_file,
        "question": question
    })
    print("AskQuestionAboutImageCamel result:", result) 