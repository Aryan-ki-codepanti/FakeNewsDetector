import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# ======== Load Tokenizer and Model ========
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


# ======== BERT Classifier Architecture ========


class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# ======== Load Trained Weights ========
print("Loading model... Please wait.")
model = BERT_Arch(bert)
model.load_state_dict(torch.load(
    "model.pt", map_location=torch.device('cpu')), strict=False)
model.eval()
print("Model loaded successfully.\n")

# ======== Input and Prediction ========


def predict_news(text):
    # Tokenize input
    tokens = tokenizer.encode_plus(
        text,
        max_length=15,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )

    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.exp(output)  # Convert log-softmax back to probabilities
        confidence, prediction = torch.max(probs, dim=1)
        label = "FAKE" if prediction.item() == 1 else "TRUE"

    return label, confidence.item()


# ======== CLI Loop ========
if __name__ == "__main__":
    print("🔍 Fake News Detector", flush=True)
    while True:
        user_input = input("\nEnter news headline (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        label, conf = predict_news(user_input)
        print(f"🔎 Prediction: {label} ({conf*100:.2f}% confidence)")
