from transformers import FlaxBertForSequenceClassification, AutoTokenizer

def main():
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_path = "/home/haoqi.whq/playground/MPCFormer/src/main/tmp/distill/CoLA/quad_2quad/bert-base-uncased/5e-05_1e-05_32_stage2"
    model_path = "/home/haoqi.whq/playground/MPCFormer/src/baselines/tmp/exp/CoLA/bert-base-uncased/5e-05"
    model = FlaxBertForSequenceClassification.from_pretrained(model_path, from_pt=True)
    print(model)
    pass # acc evaluation


if __name__ == '__main__':
    main()