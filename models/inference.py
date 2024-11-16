import torch
from core.logger import logger
from core.config import settings
from tokenizers import Tokenizer
from models.model import MtTransformerModel
from models.preprocessing import ParallelTextDataset, get_tokenizer


class MtInferenceEngine:
    
    def __init__(self, model: MtTransformerModel, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, top_k: int= 5, nucleus_threshold=10) -> None:
        self.model = model
        self.top_k = top_k
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.nucleus_threshold = nucleus_threshold
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64, device=settings.DEVICE)  # (1,)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64, device=settings.DEVICE)  # (1,)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64, device=settings.DEVICE)  # (1,)
        self.model.eval()
       
    @torch.no_grad() 
    def translate(self, source_text: str, max_len: int) -> str:
        dataset = ParallelTextDataset(
            dataset=[{"en": source_text, "am":"" }], 
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer
        )
        batch_iterator = iter( dataset.batch_iterator(1))
        batch = next(batch_iterator)
        
        encoder_input = batch["encoder_input"].to(settings.DEVICE)       # (1, seq_len) 
        encoder_mask = batch["encoder_mask"].to(settings.DEVICE)         # (1, 1, 1, seq_len) 
        decoder_mask = batch["decoder_mask"].to(settings.DEVICE)         # (1, 1, seq_len, seq_len) 
                        
        # yield self.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len)
        sos_idx = self.tgt_tokenizer.token_to_id('[SOS]')
        eos_idx = self.tgt_tokenizer.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        
        # Initialize the decoder input with the sos token
        next_token = sos_idx
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(settings.DEVICE)
        while decoder_input.size(1) < max_len and next_token != eos_idx:
            # Build required masking for decoder input
            decoder_mask = ParallelTextDataset.lookback_mask(decoder_input.size(1)).type_as(encoder_mask).to(settings.DEVICE)

            # Calculate output of decoder
            decoder_out = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)       # (1, seq_len, d_model)
            
            # Retrieve the embedded vector form of the last token
            last_token_vec = decoder_out[:, -1]                         # (1, d_model)
            
            # Get the model's raw output(logits)
            last_token_logits = self.model.project(last_token_vec)           # (1, d_model) --> (1, tgt_vocab_size)
            
            # Evaluate the probability distribution across the vocab_size 
            # dimension using softmax
            last_token_prob = torch.softmax(last_token_logits, dim=1)
            
            # Greedily pick the one with the highest probability
            _, next_token = torch.max(last_token_prob, dim=1)
            
            # Append to the decoder input for the subsequent iterations
            decoder_input = torch.cat([
                decoder_input, 
                torch.empty(1, 1).type_as(encoder_input).fill_(next_token.item()).to(settings.DEVICE)
            ], dim=1)

        # Remove the batch dimension 
        decoder_input = decoder_input.squeeze(0)                                    # torch.tensor([...]) with shape tensor.Size([max_len])
        return self.tgt_tokenizer.decode(decoder_input.detach().cpu().tolist())
    


state = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE)

src_tokenizer: Tokenizer = get_tokenizer(settings.SRC_LANG, "tokenizer-en-v3.5-20k.json")
tgt_tokenizer: Tokenizer = get_tokenizer(settings.TGT_LANG, "tokenizer-am-v3.5-20k.json")

model = MtTransformerModel.build(
    src_vocab_size=settings.VOCAB_SIZE, 
    tgt_vocab_size=settings.VOCAB_SIZE
).to(settings.DEVICE)
model.load_state_dict(state["model_state_dict"])

model.eval()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Initiating Inference on `{settings.DEVICE}` device with a model that has {params} trainable parameters.")

TRANSLATOR = MtInferenceEngine(model, src_tokenizer, tgt_tokenizer)