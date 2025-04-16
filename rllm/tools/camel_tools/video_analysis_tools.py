try:
    from camel.toolkits import VideoAnalysisToolkit
    from camel.toolkits.function_tool import FunctionTool
except ImportError as e:
    print(e)

from rllm.tools.camel_tools.camel_tool_base import CamelTool

class AskQuestionAboutVideoCamel(CamelTool):
    def __init__(self):
        super().__init__(
            function_tool=FunctionTool(VideoAnalysisToolkit().ask_question_about_video)
        )

if __name__ == "__main__":
    # issue with youtube detecting bot activity.
    
    video_file = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    question = "What is happening in this video?"
    

    print("Testing AskQuestionAboutVideoCamel")
    print("Video file:", video_file)
    print("Question:", question)
    try:
        result = AskQuestionAboutVideoCamel()(**{
            "video_path": video_file,
            "question": question
        })
        print("AskQuestionAboutVideoCamel result:", result)
    except Exception as e:
        print("Error testing AskQuestionAboutVideoCamel:", str(e)) 