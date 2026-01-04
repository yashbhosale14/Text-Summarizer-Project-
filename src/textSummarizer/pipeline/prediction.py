from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline


class PredictionPipeline:
    def __init__(self):
        # keep config exactly as is
        self.config = ConfigurationManager().get_model_evaluation_config()

        # load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        # load pipeline once (important for performance)
        self.pipe = pipeline(
            "summarization",
            model=self.config.model_path,
            tokenizer=self.tokenizer
        )

    def predict(self, text):
        # clean input text
        clean_text = text.replace("\n", " ").replace("\r", " ")

        # improved generation parameters
        gen_kwargs = {
    "max_length": 100,
    "min_length": 25,
    "num_beams": 4,
    "repetition_penalty": 2.2,
    "length_penalty": 1.1,
    "early_stopping": True,
    "no_repeat_ngram_size": 3
}


        print("Dialogue:")
        print(clean_text)

        output = self.pipe(clean_text, **gen_kwargs)[0]["summary_text"]

        # clean model artifacts like <n>
        output = (
    output
    .replace("<n>", " ")
    .replace("Eric:", "")
    .replace("Rob:", "")
    .replace('"', "")
    .replace("  ", " ")
    .strip()
)


        print("\nModel Summary:")
        print(output)

        return output
