import torch
import torch.utils.checkpoint
from torch import nn
import pickle
import os
from src.models.modules.attention import LabelAttention, MultiSynonymsAttention,LabelCrossAttention
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from transformers import BertModel, AutoConfig, AutoTokenizer, RobertaModel
class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str, scale: float = 1.0, **kwargs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_path, num_labels=num_classes, finetuning_task=None)
        self.roberta = RobertaModel(self.config, add_pooling_layer=False).from_pretrained(model_path,
                                                                                          config=self.config).to(
            self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not os.path.exists("/path/to/Synonyms.pkl"):
            with open("/path/to/Original_Synonyms.pkl", "rb") as f:
                code_synonms =  pickle.load(f)
            synonym_data = code_synonms
            print("Synonym data loaded successfully:", len(synonym_data))
            self.label_tokens = []
            self.synonyms_matrix = []
            self.roberta.eval()
            with torch.no_grad():
                for synonyms in synonym_data:
                    synonyms = list(synonyms)
                    inputs = self.tokenizer(
                        synonyms, return_tensors="pt", padding='max_length', truncation=True, max_length=512
                    )["input_ids"].to(self.device)
                    a = self.tokenizer(
                        synonyms, return_tensors="pt", padding='max_length', truncation=True, max_length=512
                    )["attention_mask"].to(self.device)
                    outputs = self.roberta(input_ids=inputs, attention_mask=a, return_dict=False)[0]
                    print(outputs.shape)
                    self.label_tokens.append(inputs)
                    out = self.Qlabel_pooling(outputs)
                    self.synonyms_matrix.append(out.clone())

                    torch.cuda.empty_cache()

            self.synonyms_matrix = pad_sequence(self.synonyms_matrix, batch_first=True, padding_value=0).to(self.device)
            self.synonyms_vector = self.qllabel_pooling(self.synonyms_matrix)
            print("SYNOMS MATRIX SHAPE", self.synonyms_matrix.shape)
            print("SYNOMS VECTOR SHAPE", self.synonyms_vector.shape)
            with open("/path/to/Synonyms-matrix.pkl", "wb") as f:
                pickle.dump(self.synonyms_matrix, f)
        else:
            with open("/path/to/Synonyms-matrix.pkl", "rb") as f:
                self.synonyms_matrix = pickle.load(f).to(self.device)
            self.synonyms_vector = self.qllabel_pooling(self.synonyms_matrix)

        self.attention_model = MultiSynonymsAttention(d_model=self.config.hidden_size, n_heads=M)


        self.label_attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        self.label_cattention = LabelCrossAttention(
            input_size=self.config.hidden_size, num_classes=num_classes, scale=scale
        )

        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.num_chunks = self.config.max_length//self.config.hidden_size
        self.chunk_size = 512
        self.hidden_size = self.config.hidden_size

        self.conv1 = nn.Conv1d(
                in_channels=3,
                out_channels=self.config.hidden_size,
                kernel_size=1
                )
        self.LeakyRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(
                in_channels=self.config.hidden_size,
                out_channels=1,
                kernel_size=1
                )
    def qllabel_pooling(self, synonyms_matrix):

        return nn.AvgPool1d(synonyms_matrix.shape[1], stride=synonyms_matrix.shape[1])(
            synonyms_matrix.permute(0, 2, 1)
        ).squeeze(2)

    def Qlabel_pooling(self,label_ids):
        return label_ids[:,0,:]

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None):
        batch_size, num_chunks, chunk_size = input_ids.size()
        device = torch.cuda.current_device()

        input_ids = input_ids.view(-1, chunk_size)

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, chunk_size)

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask if attention_mask is not None else None,
                )

        hidden_output2 = outputs[0].view(batch_size, num_chunks * chunk_size, -1)

        synonyms_attention_output = self.attention_model(
            Q=self.synonyms_matrix,
            H=hidden_output2,
            ql=self.synonyms_vector
        )

        logits1 = synonyms_attention_output
        logits1 = logits1.view(batch_size, -1, logits1.shape[-1]).max(dim=1).values
        lable_attention_output = self.label_attention(hidden_output2)
        logits2 = lable_attention_output
        label_cattention_output = self.label_cattention(hidden_output2)
        logits3 = label_cattention_output

        logits_stack1 = torch.stack([logits1, logits2, logits3], dim=1).to(device)
        x = self.conv1(logits_stack1)
        x = self.LeakyRelu(x)
        x = self.conv2(x)
        final_logits = x.squeeze(1)
        return final_logits


    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data = batch.data.to(self.device)
        targets = batch.targets.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        logits = self(data, attention_mask).to(self.device)
        loss = self.get_loss(logits, targets)

        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data = batch.data.to(self.device)
        targets = batch.targets.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        logits = self(data, attention_mask).to(self.device)
        loss = self.get_loss(logits, targets)

        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}