from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import plugins

plugins.retrieval.enable=True
plugins.retrieval.args["input_path"]="/home//Repository/AI_Coach/_data.pdf"
config = PipelineConfig(plugins=plugins)
chatbot = build_chatbot(config)

text = chatbot.predict("Hello, how are you?")
print(text)
